import torch
from pathlib import Path
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
import numpy as np

class YOLOV11Validator:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
        self.csv = self.save_dir / "results.csv"

        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        
        self.class_map = list(range(len(model.names)))
        self.names = model.names
        self.nc = len(model.names)
        self.conf = 0.001
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf)
        
        self.csv_header = [
            "epoch", "time",
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "lr/pg0", "lr/pg1", "lr/pg2"
        ]
    
    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=bbox.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn
    
    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )
    
    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    
    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def save_metrics(self, metrics, epoch, train_losses=None, val_losses=None, time_epoch=None, learning_rates=None):
        """Saves training metrics to a CSV file."""
        s = "" if self.csv.exists() else (",".join(self.csv_header) + "\n")
        
        # Prepare metrics row
        row = {
            "epoch": epoch + 1,
            "time": time_epoch or 0,
            "train/box_loss": train_losses[0] if train_losses else 0,
            "train/cls_loss": train_losses[1] if train_losses else 0,
            "train/dfl_loss": train_losses[2] if train_losses else 0,
            "metrics/precision(B)": metrics.get("metrics/precision(B)", 0),
            "metrics/recall(B)": metrics.get("metrics/recall(B)", 0),
            "metrics/mAP50(B)": metrics.get("metrics/mAP50(B)", 0),
            "metrics/mAP50-95(B)": metrics.get("metrics/mAP50-95(B)", 0),
            "val/box_loss": val_losses[0] if val_losses else 0,
            "val/cls_loss": val_losses[1] if val_losses else 0,
            "val/dfl_loss": val_losses[2] if val_losses else 0,
            "lr/pg0": learning_rates[0] if learning_rates else 0,
            "lr/pg1": learning_rates[1] if learning_rates else 0,
            "lr/pg2": learning_rates[2] if learning_rates else 0,
        }
        
        # Write row to CSV
        with open(self.csv, "a") as f:
            f.write(s + ",".join(f"{row[k]:.5g}" for k in self.csv_header) + "\n")

    def get_metrics(self, preds, batch):
        """Metrics."""
        device = preds[0].device if len(preds) > 0 else batch["img"].device
        stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        
        for si, pred in enumerate(preds):
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=device),
                pred_cls=torch.zeros(0, device=device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=device),
            )
            
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            
            if npr == 0:
                if nl:
                    for k in stats.keys():
                        stats[k].append(stat[k])
                    self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                # Update confusion matrix with both predictions and ground truth
                self.confusion_matrix.process_batch(detections=predn, gt_bboxes=bbox, gt_cls=cls)
           
            for k in stats.keys():
                stats[k].append(stat[k])
        
        # Move all tensors to CPU before concatenating and converting to numpy
        stats = {k: torch.cat([t.to(device) for t in v], 0).cpu().numpy() 
                for k, v in stats.items()}
        
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.metrics.confusion_matrix = self.confusion_matrix.matrix
        return self.metrics