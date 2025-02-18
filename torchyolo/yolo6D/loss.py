import torch.nn as nn
import torch
import torch.nn.functional as F
from ultralytics.utils.tal import make_anchors
from pipelearn.models.yolo6D.tal import TaskAlignedAssigner, dist2bbox, bbox2dist
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import bbox_iou
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from pipelearn.models.yolo6D.utils import rotation_decode, translation_decode, rotation_encode

class DFLoss(nn.Module):
    """Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Compute IoU and DFL losses for bounding boxes."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        # IoU loss
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class ADDsLoss(nn.Module):
    """ADD(-S) Loss for 6D pose estimation, grouped by class for partial vectorization."""

    def __init__(self, class_id_to_model_dict, max_points=1000):
        super().__init__()
        self.class_id_to_model_dict = class_id_to_model_dict
        self.cached_model_points = {}
        self.max_points = max_points

    def load_model_points(self, class_id, device):
        """Load normalized model points for the class, caching them."""
        if class_id in self.cached_model_points:
            return self.cached_model_points[class_id]

        # Load model
        model_dir_path = "./datasets/lmo/models"
        model_path = self.class_id_to_model_dict[str(class_id)]["file_name"]
        plydata = PlyData.read(f"{model_dir_path}/{model_path}")
        vertex = plydata["vertex"]
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
        points = torch.from_numpy(points).float()

        # Subsample points if needed
        if points.shape[0] > self.max_points:
            step = points.shape[0] // self.max_points
            points = points[::step]

        # Normalize to unit sphere
        center = points.mean(dim=0, keepdim=True)
        scale = torch.norm(points - center, dim=1).max()
        points = (points - center) / scale

        points = points.to(device)
        self.cached_model_points[class_id] = points
        return points

    def forward(
        self,
        pred_rot_matrix,    # (B, N, 3, 3)
        pred_trans,         # (B, N, 3)
        target_rot_matrix,  # (B, N, 3, 3)
        target_trans,       # (B, N, 3)
        fg_mask,            # (B, N)
        class_ids,          # (B, N)
    ):
        device = pred_rot_matrix.device
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Get foreground indices
        # Flatten to (M,) where M = number of FG instances
        fg_indices = fg_mask.nonzero(as_tuple=False)  # (M, 2) -> (batch_idx, obj_idx)

        # Extract FG data
        b_idx = fg_indices[:, 0]
        n_idx = fg_indices[:, 1]

        fg_pred_R = pred_rot_matrix[b_idx, n_idx]      # (M, 3, 3)
        fg_pred_t = pred_trans[b_idx, n_idx]           # (M, 3)
        fg_gt_R = target_rot_matrix[b_idx, n_idx]      # (M, 3, 3)
        fg_gt_t = target_trans[b_idx, n_idx]           # (M, 3)
        fg_classes = class_ids[b_idx, n_idx]           # (M,)

        # Group instances by class
        unique_classes = fg_classes.unique()
        total_loss = 0.0
        total_instances = fg_classes.shape[0]

        with torch.no_grad():
            for c in unique_classes:
                # Get all instances of this class
                mask_class = (fg_classes == c)
                class_indices = mask_class.nonzero(as_tuple=False).squeeze(1)

                # Load model points for this class
                points = self.load_model_points(int(c.item()), device)  # (P, 3)
                P = points.shape[0]

                # Gather rotations and translations for this class
                class_pred_R = fg_pred_R[class_indices]   # (M_c, 3, 3)
                class_pred_t = fg_pred_t[class_indices]   # (M_c, 3)
                class_gt_R = fg_gt_R[class_indices]       # (M_c, 3, 3)
                class_gt_t = fg_gt_t[class_indices]       # (M_c, 3)

                M_c = class_pred_R.shape[0]

                # Expand points for batch processing:
                # points: (P, 3) -> (M_c, P, 3)
                points_expanded = points.unsqueeze(0).expand(M_c, P, 3)

                # Transform model points with predicted pose
                # pred_pts[i] = pred_R[i] @ points.T + pred_t[i]
                # pred_R[i]: (3,3), points: (M_c, P, 3)
                # Need to do (M_c,3,3) @ (M_c,3,P) -> (M_c,3,P) then transpose -> (M_c,P,3)
                pred_pts = torch.matmul(class_pred_R, points_expanded.transpose(1,2)).transpose(1,2) + class_pred_t.unsqueeze(1)

                # Transform model points with ground truth pose
                gt_pts = torch.matmul(class_gt_R, points_expanded.transpose(1,2)).transpose(1,2) + class_gt_t.unsqueeze(1)

                # Flatten points for KD-tree:
                # pred_pts: (M_c, P, 3) -> (M_c*P, 3)
                # gt_pts: (M_c, P, 3) -> (M_c*P, 3)
                pred_pts_flat = pred_pts.reshape(M_c*P, 3).cpu().numpy()
                gt_pts_flat = gt_pts.reshape(M_c*P, 3).cpu().numpy()

                # Build KD-trees
                kdtree_gt = cKDTree(gt_pts_flat)
                dist_pred_to_gt, _ = kdtree_gt.query(pred_pts_flat, k=1)

                kdtree_pred = cKDTree(pred_pts_flat)
                dist_gt_to_pred, _ = kdtree_pred.query(gt_pts_flat, k=1)

                # Compute ADD-S for each instance within this class
                # Split distances back into (M_c, P)
                dist_pred_to_gt_split = dist_pred_to_gt.reshape(M_c, P)
                dist_gt_to_pred_split = dist_gt_to_pred.reshape(M_c, P)

                # Average symmetrical distance per instance
                adds_per_instance = 0.5 * (dist_pred_to_gt_split.mean(axis=1) + dist_gt_to_pred_split.mean(axis=1))

                # Sum over instances of this class
                total_loss += adds_per_instance.sum()

        # Compute the average over all foreground instances
        return torch.tensor(total_loss / total_instances, device=device)

class OKSLoss(nn.Module):
    """Object Keypoint Similarity (OKS) Loss.

    Object area can be provided directly or calculated from the bounding box.
    """

    def __init__(self, k=0.1):
        super().__init__()
        self.k = k

    def forward(
        self,
        pred_trans,
        target_trans,
        object_area,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        if fg_mask.sum():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            area = object_area[fg_mask]

            # Distance between predicted and ground truth (cx, cy)
            d = torch.norm(pred_trans[fg_mask][:, :2] - target_trans[fg_mask][:, :2], dim=1)

            # Compute OKS (object keypoint similarity)
            oks = torch.exp(-(d**2) / (2 * area**2 * self.k**2))

            # Compute weighted loss
            return ((1 - oks) * weight.squeeze(-1)).sum() / target_scores_sum

        return torch.tensor(0.0, device=pred_trans.device)

class ARDLoss(nn.Module):
    """Absolute Relative Difference (ARD) Loss for Depth."""

    def forward(self, pred_tz, target_tz, target_scores, target_scores_sum, fg_mask):
        if fg_mask.sum():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            ard = torch.abs(1 - pred_tz[fg_mask] / target_tz[fg_mask])
            return (ard * weight.squeeze(-1)).sum() / target_scores_sum

        return torch.tensor(0.0, device=pred_tz.device)


class RotationLoss(nn.Module):
    """Geodesic rotation loss for 3x3 rotation matrix."""

    def forward(
        self,
        pred_rot_matrix,
        target_rot_matrix,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        if fg_mask.sum():
            weight = target_scores.sum(-1)[fg_mask]

            # Compute R1 * R2.T (matrix difference)
            R_diff = torch.matmul(
                pred_rot_matrix[fg_mask], target_rot_matrix[fg_mask].transpose(-2, -1)
            )

            # Calculate the rotation angle using matrix trace
            trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
            cos_theta = (trace - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            theta = torch.acos(cos_theta)

            return (theta * weight).sum() / target_scores_sum

        return torch.tensor(0.0, device=pred_rot_matrix.device)

class CombinedPoseLoss(nn.Module):
    """Combined Pose Loss for 6D Pose Estimation."""

    def __init__(
        self,
        model,
        class_id_to_model_dict,
        tal_topk=10,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
        rot_weight=2.0,
        oks_weight=1.0,
        ard_weight=1.0,
        adds_weight=1.0,
        k=0.1,
    ):
        super().__init__()
        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + (m.reg_max * 4) + 6 + 3
        self.reg_max = m.reg_max

        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.rot_weight = rot_weight
        self.oks_weight = oks_weight
        self.ard_weight = ard_weight
        self.adds_weight = adds_weight
        self.k = k
        self.class_id_to_model_dict = class_id_to_model_dict

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max)
        self.adds_loss = ADDsLoss(self.class_id_to_model_dict)
        self.rotation_loss = RotationLoss()
        self.oks_loss = OKSLoss(k=k)
        self.ard_loss = ARDLoss()

        self.register_buffer("proj", torch.arange(m.reg_max, dtype=torch.float))

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses targets."""
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, 14, device=targets.device)

        i = targets[:, 0]  # Image indices
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)

        out = torch.zeros(batch_size, counts.max(), 14, device=targets.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                # Class labels
                out[j, :n, 0] = targets[matches, 1]

                # Bboxes
                bboxes = targets[matches, 2:6]
                out[j, :n, 1:5] = xywh2xyxy(bboxes.mul(scale_tensor))

                # Encode rotation matrix to 6D representation
                rotation_matrix = targets[matches, 6:15].view(-1, 3, 3)
                rot_6d = rotation_encode(rotation_matrix)
                out[j, :n, 5:11] = rot_6d

                # cx, cy (scale to pixel coordinates)
                cx_cy = targets[matches, 15:17]
                cx_cy[:, 0] = cx_cy[:, 0] * scale_tensor[0]
                cx_cy[:, 1] = cx_cy[:, 1] * scale_tensor[1]
                out[j, :n, 11:13] = cx_cy

                # tz
                out[j, :n, 13] = targets[matches, 17]

        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode bbox distribution to xyxy."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds, batch):
        device = preds[0].device if isinstance(preds, (tuple, list)) else preds.device
        loss = torch.zeros(7, device=device)  # box, cls, dfl, adds, rot, oks, ard
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_bbox_distri, pred_class, pred_rot, pred_trans = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc, 6, 3), 1)

        pred_bbox_distri = pred_bbox_distri.permute(0, 2, 1).contiguous()
        pred_class = pred_class.permute(0, 2, 1).contiguous()
        pred_rot = pred_rot.permute(0, 2, 1).contiguous()
        pred_trans = pred_trans.permute(0, 2, 1).contiguous()

        dtype = pred_class.dtype
        batch_size = pred_class.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=device, dtype=dtype)
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Prepare targets
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["labels"].view(-1, 1),
                batch["bboxes"],
                batch["rotations"].view(-1, 9),
                batch["cx_cys"],
                batch["tzs"],
            ),
            1,
        ).to(device)

        targets = self.preprocess(
            targets,
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],  # [w, h, w, h]
        )

        # Split targets
        gt_labels, gt_bboxes, gt_rotations, gt_translations = targets.split(
            (1, 4, 6, 3), dim=2
        )
        mask_gt = gt_bboxes.sum(dim=2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_bbox_distri)  # xyxy

        # Assign Targets
        (
            target_labels,
            target_bboxes,
            target_rotations,
            target_translations,
            target_scores,
            fg_mask,
            class_ids,
        ) = self.assigner.forward(
            pred_class.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            pred_rot,
            pred_trans,
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_rotations,
            gt_translations,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss
        loss[1] = (
            self.bce(pred_class, target_scores.to(dtype)).sum() / target_scores_sum
        )

        # Bbox and DFL loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_bbox_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # Decode rotations
            pred_rot_matrix = rotation_decode(pred_rot)
            target_rot_matrix = rotation_decode(target_rotations)

            # Decode translations (for predictions)
            pred_trans_decoded = translation_decode(
                pred_trans, anchor_points, stride_tensor
            )
            # target_translations are already (cx, cy, tz) in pixel coords from preprocess
            target_trans_decoded = target_translations  # (B, N, 3)

            # Not using ADD(-S) loss since the implementation could be incorrect
            # ADD(-S) loss
            # loss[3] = self.adds_loss(
            #     pred_rot_matrix,
            #     pred_trans_decoded,
            #     target_rot_matrix,
            #     target_trans_decoded,
            #     fg_mask,
            #     target_labels,
            # )
            loss[3] = torch.tensor(0.0, device=pred_trans_decoded.device)

            # Rotation loss
            loss[4] = self.rotation_loss(
                pred_rot_matrix,
                target_rot_matrix,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # loss[4] = torch.tensor(0.0, device=pred_rot_matrix.device)

            og_bboxes = target_bboxes * stride_tensor
            bbox_area = (og_bboxes[:, :, 2] - og_bboxes[:, :, 0]) * (og_bboxes[:, :, 3] - og_bboxes[:, :, 1])
            # OKS loss
            loss[5] = self.oks_loss(
                pred_trans_decoded,
                target_trans_decoded,
                bbox_area,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # loss[5] = torch.tensor(0.0, device=pred_trans_decoded.device)

            # ARD loss (depth comparison)
            loss[6] = self.ard_loss(
                pred_trans_decoded[:, :, 2],  # predicted tz
                target_trans_decoded[:, :, 2],  # target tz
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # loss[6] = torch.tensor(0.0, device=pred_trans_decoded.device)

        # Apply weights
        loss[0] *= self.box_weight
        loss[1] *= self.cls_weight
        loss[2] *= self.dfl_weight
        loss[3] *= self.adds_weight
        loss[4] *= self.rot_weight
        loss[5] *= self.oks_weight
        loss[6] *= self.ard_weight

        return loss.sum() * batch_size, loss.detach()
