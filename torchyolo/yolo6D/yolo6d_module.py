from embdata.geometry import Transform3D
import pytorch_lightning as pl
from .yolo6d_model import YOLO6DModel
import torch
from pipelearn.models.yolo6D.loss import CombinedPoseLoss
from torch import nn, optim
from pathlib import Path
from ultralytics.utils import ops
from datetime import datetime
from copy import deepcopy
import time
from ultralytics.utils.ops import xywh2xyxy
import json
from tqdm import tqdm
import math
import numpy as np
from ultralytics.utils import ops
from typing import Dict
from pipelearn.models.yolo6D.utils import (
    box_iou,
    project_translation,
    build_camera_matrix,
)


class YOLO6DModule(pl.LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        save_dir: str,
        freeze_backbone: bool = False,
        override_mapping: Dict[int, str] = None,
        class_id_to_model_json: Dict[int, str] = None,
        # Learning rate parameters
        lr0: float = 0.01,  # Initial learning rate
        lrf: float = 0.01,  # Final learning rate fraction
        momentum: float = 0.937,  # SGD momentum/Adam beta1
        weight_decay: float = 0.0005,  # Optimizer weight decay
        # Scheduler parameters
        warmup_steps: float = 100,  # Changed to float as per args.yaml
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        cos_lr: bool = False,  # Use cosine LR scheduler
        # Training parameters
        save_every: int = 500,
        batch_size: int = 32,
        use_lr_scheduler: bool = True,
        # Loss weights (optional, but good to have from args.yaml)
        rot_weight: float = 1.5,
        cls_weight: float = 0.5,
        adds_weight: float = 1.5,
        oks_weight: float = 1.0,
        ard_weight: float = 1.0,
    ):
        super().__init__()
        self.model = YOLO6DModel(
            ckpt_path=ckpt_path,
            freeze_backbone=freeze_backbone,
            override_mapping=override_mapping,
        )
        self.criterion = CombinedPoseLoss(
            self.model,
            class_id_to_model_dict=class_id_to_model_json,
            rot_weight=rot_weight,
            cls_weight=cls_weight,
            adds_weight=adds_weight,
            oks_weight=oks_weight,
            ard_weight=ard_weight,
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Store all hyperparameters
        self.lr0 = lr0
        self.lrf = lrf
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.cos_lr = cos_lr
        self.save_every = save_every
        self.batch_size = batch_size
        self.use_lr_scheduler = use_lr_scheduler

        # Initialize tracking variables
        self.running_epoch = 0
        self.nb = 0  # Will store number of batches

    def training_step(self, batch, batch_idx):

        # Normalize images and move to device
        batch["image"] = batch["image"].to(self.device, non_blocking=True).float() / 255
        for k in ["batch_idx", "labels", "bboxes", "tzs", "rotations", "cx_cys"]:
            batch[k] = batch[k].to(self.device)

        # Forward pass
        preds = self.model(batch["image"])
        loss, individual_losses = self.criterion(preds, batch)

        # Log losses with sync_dist=True
        bbox_loss, cls_loss, dfl_loss, adds_loss, rot_loss, oks_loss, ard_loss = (
            individual_losses
        )
        self.log("train_bbox_loss", bbox_loss.item(), sync_dist=True)
        self.log("train_cls_loss", cls_loss.item(), sync_dist=True)
        self.log("train_dfl_loss", dfl_loss.item(), sync_dist=True)
        self.log("train_adds_loss", adds_loss.item(), sync_dist=True)
        self.log("train_rot_loss", rot_loss.item(), sync_dist=True)
        self.log("train_oks_loss", oks_loss.item(), sync_dist=True)
        self.log("train_ard_loss", ard_loss.item(), sync_dist=True)

        if (
            self.global_rank == 0
            and self.trainer.global_step > 0
            and self.trainer.global_step % self.save_every == 0
        ):
            self.save_model(f"step_{self.trainer.global_step}.pt")

        return loss

    def on_after_backward(self):
        # First measure the gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"))
        self.log("grad_norm", grad_norm, sync_dist=True)

        # Then actually clip if needed
        if grad_norm > 10.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler matching Ultralytics implementation."""
        # Parameter groups setup
        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers

        # Separate parameters into groups
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # biases
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        # Get number of batches from trainer
        iterations = self.trainer.estimated_stepping_batches
        # Select optimizer and initial learning rate
        if iterations > 10000:
            optimizer = optim.SGD(
                g[2], lr=self.lr0, momentum=self.momentum, nesterov=True
            )
        else:
            nc = getattr(self.model, "nc", 10)  # number of classes
            self.lr0 = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation
            optimizer = optim.AdamW(g[2], lr=self.lr0, betas=(self.momentum, 0.999))
            self.warmup_bias_lr = 0.0  # Prevent warmup for shorter trainings with Adam

        # Add parameter groups
        optimizer.add_param_group({"params": g[0], "weight_decay": self.weight_decay})
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

        # Store initial learning rates for each group
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        # Store optimizer instance
        self.optimizer = optimizer

        if not self.use_lr_scheduler:
            return optimizer

        def one_cycle(y1=0.0, y2=1.0, steps=100):
            """Returns a lambda function for sinusoidal learning rate decay from y1 to y2 over steps."""
            return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

        # Get total steps from trainer
        total_steps = self.trainer.estimated_stepping_batches

        # Setup scheduler
        if self.cos_lr:
            self.lf = one_cycle(1, self.lrf, total_steps)
        else:
            self.lf = (
                lambda x: (1 - x / total_steps) * (1.0 - self.lrf) + self.lrf
            )  # linear

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Changed from 'epoch' to 'step'
                "frequency": 1,
            },
        }

    def validation_step(self, batch, batch_idx):
        # Normalize images and move to device
        batch["image"] = batch["image"].to(self.device).float() / 255
        for k in ["batch_idx", "labels", "bboxes", "tzs", "rotations", "cx_cys"]:
            batch[k] = batch[k].to(self.device)

        preds = self.model(batch["image"])
        val_loss, individual_losses = self.criterion(preds, batch)
        (
            bbox_loss,
            cls_loss,
            dfl_loss,
            adds_loss,
            rot_loss,
            oks_loss,
            ard_loss,
        ) = individual_losses

        self.log("val_bbox_loss", bbox_loss.item(), sync_dist=True)
        self.log("val_cls_loss", cls_loss.item(), sync_dist=True)
        self.log("val_dfl_loss", dfl_loss.item(), sync_dist=True)
        self.log("val_adds_loss", adds_loss.item(), sync_dist=True)
        self.log("val_rot_loss", rot_loss.item(), sync_dist=True)
        self.log("val_oks_loss", oks_loss.item(), sync_dist=True)
        self.log("val_ard_loss", ard_loss.item(), sync_dist=True)

        return val_loss

    def save_model(self, filename="model.pt"):
        """Save model checkpoint with metadata."""
        save_path = self.save_dir / "weights"
        save_path.mkdir(parents=True, exist_ok=True)

        # Create a copy of the model and attach the yaml config
        model_copy = deepcopy(self.model).half()
        model_copy.yaml = (
            self.model.model_cfg
        )  # Attach yaml config to model before saving

        # Prepare checkpoint data
        ckpt = {
            "epoch": self.current_epoch,
            "model": model_copy,  # Save model with attached yaml config
            "optimizer": self.trainer.optimizers[0].state_dict(),
            "train_args": {
                "lr0": self.lr0,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
            },
            "date": datetime.now().isoformat(),
        }

        # Save the checkpoint
        torch.save(ckpt, save_path / filename)
        print(f"Saved checkpoint to {save_path / filename}")

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Args:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + 9 + 3 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels.

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 18 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, r1, r2, r3, r4, r5, r6, r7, r8, r9, tx, ty, tz, mask1, mask2, ...).
        """
        import torchvision

        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(prediction, (list, tuple)):  # output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        if classes is not None:
            classes = torch.tensor(classes, device=prediction.device)

        bs = prediction.shape[0]
        nc = nc or (prediction.shape[1] - (9 + 3 + 4))  # number of classes
        nm = prediction.shape[1] - nc - (9 + 3 + 4)  # number of masks
        mi = (9 + 3 + 4) + nc  # mask start index
        xc = prediction[:, 4 : 4 + nc].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 2.0 + max_time_img * bs  # seconds to quit after

        prediction = prediction.transpose(-1, -2)

        prediction[..., :4] = xywh2xyxy(prediction[..., :4])

        t = time.time()
        output = [torch.zeros((0, 18 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx18 (xyxy, conf, cls, rot, trans, mask)
            box, cls, rot, trans, mask = x.split((4, nc, 9, 3, nm), 1)

            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), rot, trans, mask), 1)[
                conf.view(-1) > conf_thres
            ]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[
                    x[:, 4].argsort(descending=True)[:max_nms]
                ]  # sort by confidence and remove excess boxes

            # Batched NMS
            c: torch.Tensor = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores

            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

    def predict_postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = self.non_max_suppression(
            preds,
            conf_thres=0.001,
            iou_thres=0.7,
        )

        if not isinstance(
            orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img in zip(preds, orig_imgs):
            # pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(pred)

        return results

    def evaluate_dataloader(
        self, dataloader, iou_thres=0.5, conf_thres=0.5, save_dir=None
    ):
        """
        Evaluate model on dataloader and compute TP/FP/FN metrics for 6D pose with bounding boxes.
        Optionally saves visualization of first batch predictions if save_dir is provided.

        Args:
            dataloader: DataLoader to evaluate on
            iou_thres (float): IoU threshold for matching predictions to ground truth
            conf_thres (float): Confidence threshold for predictions
            save_dir (str, optional): Directory to save visualization images. If None, no images are saved
        """

        def update_category_stats_tp_fp_fn(
            preds: torch.Tensor,
            gts: torch.Tensor,
            category_stats: dict,
            iou_thres=0.5,
            conf_thres=0.25,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
        ):
            """
            Update per-category stats (tp, fp, fn) and compare rotation/translation for one image.

            Args:
                preds (torch.Tensor): shape (P, 18) =>
                    [x1, y1, x2, y2, conf, cls,
                    r1, r2, r3, r4, r5, r6, r7, r8, r9,
                    cx_pred, cy_pred, tz_pred]
                gts (torch.Tensor): shape (G, 17) =>
                    [cls, x1,y1,x2,y2,
                    r1,r2,r3,r4,r5,r6,r7,r8,r9,
                    cx_gt, cy_gt, tz_gt]
                category_stats (dict): {class_id: {"tp": int, "fp": int, "fn": int}}
                iou_thres (float): IoU threshold for a positive match
                conf_thres (float): confidence threshold for considering a detection
                fx, fy, cx, cy (float): Camera intrinsics
            """
            device = preds.device
            gts = gts.to(device)

            # Filter out low-confidence predictions
            if preds.numel() > 0:
                mask = preds[:, 4] >= conf_thres
                preds = preds[mask]
            else:
                preds = torch.empty((0, 18), device=device)

            # No predictions => all gts are FN
            if len(preds) == 0 and len(gts) > 0:
                for gt in gts:
                    gt_cls = int(gt[0])
                    category_stats[gt_cls]["fn"] += 1
                return

            # No ground truths => all preds are FP
            if len(gts) == 0 and len(preds) > 0:
                for det in preds:
                    det_cls = int(det[5])
                    category_stats[det_cls]["fp"] += 1
                return

            # Both are empty => do nothing
            if len(preds) == 0 and len(gts) == 0:
                return

            # IoU matching
            ious = box_iou(preds[:, :4], gts[:, 1:5])  # shape (nd, ng)
            nd, ng = ious.shape

            det_cls = preds[:, 5].long()  # predicted classes
            gt_cls = gts[:, 0].long()  # ground-truth classes

            iou_best, iou_best_idx = ious.max(
                dim=1
            )  # best IoU & idx for each detection
            matched_gt = torch.zeros(ng, dtype=torch.bool, device=device)

            # Sort detections by confidence (descending)
            conf_sort_idx = torch.argsort(preds[:, 4], descending=True)
            det_cls = det_cls[conf_sort_idx]
            iou_best = iou_best[conf_sort_idx]
            iou_best_idx = iou_best_idx[conf_sort_idx]
            preds = preds[conf_sort_idx]  # reorder preds

            # Build the inverse K
            K, K_inv = build_camera_matrix(fx, fy, cx, cy, device=device)

            for i in range(nd):
                if iou_best[i] >= iou_thres:
                    gt_i = iou_best_idx[i]
                    if not matched_gt[gt_i]:
                        if det_cls[i] == gt_cls[gt_i]:
                            matched_gt[gt_i] = True
                            category_stats[det_cls[i].item()]["tp"] += 1

                            # ============= Bbox =============
                            pred_bbox = preds[i, :4]
                            gt_bbox = gts[gt_i, 1:5]

                            # 1) Rotation (3x3)
                            pred_rot = preds[i, 6:15].view(3, 3)

                            # 2) Pred (cx, cy, tz)
                            cx_pred = preds[i, 15]
                            cy_pred = preds[i, 16]
                            tz_pred = preds[i, 17]

                            # Here, let's assume they are already pixel coords so we skip scaling:
                            cx_pred_px = cx_pred
                            cy_pred_px = cy_pred

                            # Then apply t_pred = tz_pred * K_inv @ [cx_pred_px, cy_pred_px, 1]^T
                            pred_trans = project_translation(
                                cx_pred_px, cy_pred_px, tz_pred, K_inv
                            )

                            pred_pose = Transform3D(
                                rotation=pred_rot.cpu().numpy(),
                                translation=pred_trans.cpu().numpy(),
                            ).pose()

                            # ============= Decode the ground-truth 6D pose =============
                            # 1) Rotation (3x3)
                            gt_rot = gts[gt_i, 5:14].view(3, 3)

                            # 2) GT (cx_gt, cy_gt, tz_gt)
                            cx_gt = gts[gt_i, 14]
                            cy_gt = gts[gt_i, 15]
                            tz_gt = gts[gt_i, 16]

                            gt_trans = project_translation(cx_gt, cy_gt, tz_gt, K_inv)
                            gt_pose = Transform3D(
                                rotation=gt_rot.cpu().numpy(),
                                translation=gt_trans.cpu().numpy(),
                            ).pose()

                            # ============= Translation error =============
                            per_coordinate_error = torch.abs(pred_trans - gt_trans)
                            translation_error = torch.norm(pred_trans - gt_trans)
                            category_stats[det_cls[i].item()][
                                "translation_error"
                            ] += translation_error.item()
                            category_stats[det_cls[i].item()][
                                "trans_x_error"
                            ] += per_coordinate_error[0].item()
                            category_stats[det_cls[i].item()][
                                "trans_y_error"
                            ] += per_coordinate_error[1].item()
                            category_stats[det_cls[i].item()][
                                "trans_z_error"
                            ] += per_coordinate_error[2].item()

                            # Add rotation error calculation
                            # Compute R1 * R2.T (matrix difference)
                            R_diff = torch.matmul(
                                pred_rot.to(dtype=torch.float32),
                                gt_rot.to(dtype=torch.float32).transpose(-2, -1)
                            )
                            
                            # Calculate the rotation angle using matrix trace
                            trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
                            cos_theta = (trace - 1) / 2
                            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                            theta = torch.acos(cos_theta)
                            
                            # Convert to degrees
                            rot_error_deg = theta.item() * 180.0 / np.pi
                            category_stats[det_cls[i].item()]["rotation_error"] += rot_error_deg
                        else:
                            category_stats[det_cls[i].item()]["fp"] += 1
                    else:
                        category_stats[det_cls[i].item()]["fp"] += 1

            # Unmatched ground truths => FN
            for gt_i in range(ng):
                if not matched_gt[gt_i]:
                    category_stats[gt_cls[gt_i].item()]["fn"] += 1

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        num_classes = len(self.model.names)
        category_stats = {
            cid: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "translation_error": 0,
                "trans_x_error": 0,
                "trans_y_error": 0,
                "trans_z_error": 0,
                "rotation_error": 0,
            }
            for cid in range(num_classes)
        }

        total_preds = 0
        total_gts = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating", leave=True)
            for batch_i, batch in enumerate(pbar):
                images = batch["image"].to(device).float() / 255
                preds = self.model(images)
                results = self.predict_postprocess(preds, images, batch["image"])

                bsz = images.shape[0]
                for img_idx in range(bsz):
                    current_img_mask = batch["batch_idx"] == img_idx

                    # Prepare GT
                    gt_cls = batch["labels"][current_img_mask].float()
                    gt_box = batch["bboxes"][current_img_mask].float()  # (cx, cy, w, h)
                    gt_rot = batch["rotations"][current_img_mask]
                    gt_cxcy = batch["cx_cys"][current_img_mask]  # (N,2)
                    gt_tz = batch["tzs"][current_img_mask]  # (N,)

                    # Camera intrinsics
                    gt_camera = batch["camera"][img_idx]
                    fx = gt_camera["intrinsic"]["fx"]
                    fy = gt_camera["intrinsic"]["fy"]
                    cx = gt_camera["intrinsic"]["cx"]
                    cy = gt_camera["intrinsic"]["cy"]

                    # Convert xywh->xyxy in pixel coords
                    gt_box_xyxy = ops.xywh2xyxy(gt_box)
                    gt_box_xyxy *= torch.tensor(
                        [
                            images.shape[3],
                            images.shape[2],
                            images.shape[3],
                            images.shape[2],
                        ],
                        device=gt_box_xyxy.device,
                    )

                    gt_cxcy *= torch.tensor(
                        [
                            images.shape[3],
                            images.shape[2],
                        ],
                        device=gt_cxcy.device,
                    )

                    # We store gts as: [cls, x1,y1,x2,y2, rot(9 floats), cx, cy, tz]
                    # total 1 + 4 + 9 + 3 = 17
                    #   (Important: if your dataloader is normalized, make sure (cx, cy) are in pixels before storing)
                    gts = torch.cat(
                        [
                            gt_cls,
                            gt_box_xyxy,  # shape (N,4)
                            gt_rot.reshape(-1, 9),  # shape (N,9)
                            gt_cxcy,  # shape (N,2)
                            gt_tz,  # shape (N,1)
                        ],
                        dim=1,
                    )  # shape (N, 17)

                    # Prepare preds
                    # Each row => [x1,y1,x2,y2, conf, cls,
                    #              rot(9 floats), cx,cy, tz] => total 18
                    if len(results[img_idx]) == 0:
                        pred_for_eval = torch.empty((0, 18), device=device)
                    else:
                        pred_for_eval = results[img_idx]

                    total_preds += len(pred_for_eval)
                    total_gts += len(gts)

                    # Now match them
                    update_category_stats_tp_fp_fn(
                        pred_for_eval,
                        gts,
                        category_stats,
                        iou_thres=iou_thres,
                        conf_thres=conf_thres,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )

            # Done. Print final metrics
            print(f"\nTotal predictions: {total_preds}")
            print(f"Total ground truths: {total_gts}\n")
            print("Per-class metrics =>")
            for cid in range(num_classes):
                stats = category_stats[cid]
                cls_name = self.model.names.get(str(cid), f"class_{cid}")

                tp = stats["tp"]
                fp = stats["fp"]
                fn = stats["fn"]

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

                # Calculate mean errors
                mean_trans_error = stats["translation_error"] / stats["tp"] if stats["tp"] > 0 else 0
                mean_x_error = stats["trans_x_error"] / stats["tp"] if stats["tp"] > 0 else 0
                mean_y_error = stats["trans_y_error"] / stats["tp"] if stats["tp"] > 0 else 0
                mean_z_error = stats["trans_z_error"] / stats["tp"] if stats["tp"] > 0 else 0

                # Calculate mean rotation error
                mean_rot_error = stats["rotation_error"] / stats["tp"] if stats["tp"] > 0 else 0

                stats.update({
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "f1_score": f1_score,
                    "translation_error": mean_trans_error,
                    "trans_x_error": mean_x_error,
                    "trans_y_error": mean_y_error,
                    "trans_z_error": mean_z_error,
                    "rotation_error": mean_rot_error,
                })

                print(f"{cls_name}:")
                print(f"  TP={tp}, FP={fp}, FN={fn}")
                print(f"  Precision={precision:.4f}, Recall={recall:.4f}")
                print(f"  Accuracy={accuracy:.4f}, F1-Score={f1_score:.4f}")
                print(f"  Mean Translation Error={mean_trans_error:.4f}m")
                print(f"  Mean X Error={mean_x_error:.4f}")
                print(f"  Mean Y Error={mean_y_error:.4f}")
                print(f"  Mean Z Error={mean_z_error:.4f}")
                print(f"  Mean Rotation Error={mean_rot_error:.4f}°")
                print(f" \n")
                print("================================================")

        return category_stats

    def on_train_batch_start(self, batch, batch_idx):
        """Handle warmup for learning rate and momentum."""
        ni = self.trainer.global_step
        nw = self.warmup_steps

        if ni <= nw:
            xi = [0, nw]  # x interp
            for j, x in enumerate(self.optimizer.param_groups):
                # Bias lr falls from warmup_bias_lr to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    ni,
                    xi,
                    [
                        (
                            self.warmup_bias_lr if j == 2 else 0.0
                        ),  # start lr (j==2 is for biases)
                        x["initial_lr"] * self.lf(self.current_epoch),
                    ],  # end lr
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        ni, xi, [self.warmup_momentum, self.momentum]
                    )

        # Log learning rates at start of each epoch
        if batch_idx == 0:  # Log at start of each epoch
            self.log_dict(
                {
                    f"lr/pg{i}": x["lr"]
                    for i, x in enumerate(self.optimizer.param_groups)
                },
                sync_dist=True,
            )

    def on_train_start(self):
        """Called when training starts."""
        # Store optimizer reference
        self.optimizer = self.trainer.optimizers[0]
