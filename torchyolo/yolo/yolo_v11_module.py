import pytorch_lightning as pl
from torchyolo.yolo.yolo11_model import YOLOv11Model
import torch
from torchyolo.yolo.loss import v8DetectionLoss
from torch import nn, optim
from ultralytics.utils.plotting import plot_results
from pathlib import Path
from ultralytics.utils import ops
from torchyolo.yolo.validator import YOLOV11Validator
from datetime import datetime
from copy import deepcopy
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16
import time
from ultralytics.engine.results import Results
import json
from tqdm import tqdm
import math
import numpy as np
from typing import Dict
import os


class YOLOv11Module(pl.LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        save_dir: str,
        freeze_backbone: bool = False,
        override_mapping: Dict[int, str] = None,
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
        batch_size: int = 32,  # Added from args.yaml
        use_lr_scheduler: bool = True,
        # Loss weights (optional, but good to have from args.yaml)
        box_loss_weight: float = 7.5,  # box loss weight
        cls_loss_weight: float = 0.5,  # cls loss weight
        dfl_loss_weight: float = 1.5,  # dfl loss weight
        # Profiler parameters
        profiler_config: dict = None,
    ):
        super().__init__()
        self.model = YOLOv11Model(
            ckpt_path=ckpt_path,
            freeze_backbone=freeze_backbone,
            override_mapping=override_mapping,
        )
        self.criterion = v8DetectionLoss(
            self.model,
            box_weight=box_loss_weight,
            cls_weight=cls_loss_weight,
            dfl_weight=dfl_loss_weight,
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.validator = YOLOV11Validator(model=self.model, save_dir=self.save_dir)

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
        self.profiler_enabled = profiler_config.get("enabled", False)
        self.trace_path = profiler_config.get("trace_path", "./lightning_logs/profiler")

    def training_step(self, batch, batch_idx):

        if torch.isnan(batch["img"]).any():
            print("NaN detected in input images!")
            return

        # Normalize images and move to device
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        # Forward pass
        if self.profiler_enabled and batch_idx < 5:  # Only profile first 25 batches
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.trace_path
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:
                preds = self.model(batch["img"])
                loss, individual_losses = self.criterion(preds, batch)
                profiler.step()  # Move profiler forward
        else:
            preds = self.model(batch["img"])
            loss, individual_losses = self.criterion(preds, batch)

        # Log losses with sync_dist=True
        bbox_loss, cls_loss, dfl_loss = individual_losses
        self.log("train_bbox_loss", bbox_loss.item(), sync_dist=True)
        self.log("train_cls_loss", cls_loss.item(), sync_dist=True)
        self.log("train_dfl_loss", dfl_loss.item(), sync_dist=True)

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
            self.lf = one_cycle(1, self.lrf, total_steps)  # cosine 1->lrf
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
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.profiler_enabled and batch_idx < 5:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.trace_path
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:
                preds = self.model(batch["img"])
                val_loss, individual_losses = self.criterion(preds, batch)
                profiler.step()
        else:
            preds = self.model(batch["img"])
            val_loss, individual_losses = self.criterion(preds, batch)

        bbox_loss, cls_loss, dfl_loss = individual_losses

        self.log("val_bbox_loss", bbox_loss.item(), sync_dist=True)
        self.log("val_cls_loss", cls_loss.item(), sync_dist=True)
        self.log("val_dfl_loss", dfl_loss.item(), sync_dist=True)

        return val_loss

    def on_train_end(self):
        """Called when training ends."""
        if not hasattr(self, "epoch_start_time"):
            self.epoch_start_time = time.time()
        epoch_time = time.time() - self.epoch_start_time

        # Get current learning rates
        lrs = [pg["lr"] for pg in self.trainer.optimizers[0].param_groups]

        # Calculate average training losses from the last epoch
        train_losses = [
            self.trainer.callback_metrics.get("train_bbox_loss").item(),
            self.trainer.callback_metrics.get("train_cls_loss").item(),
            self.trainer.callback_metrics.get("train_dfl_loss").item(),
        ]

        # Get validation losses
        val_losses = [
            self.trainer.callback_metrics.get("val_bbox_loss").item(),
            self.trainer.callback_metrics.get("val_cls_loss").item(),
            self.trainer.callback_metrics.get("val_dfl_loss").item(),
        ]

        # Set model to eval mode
        self.model.eval()

        # Calculate metrics using validation data
        with torch.no_grad():
            for batch in self.trainer.datamodule.val_dataloader():
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                batch["img"] = batch["img"].float() / 255
                # Get predictions
                preds = self.model(batch["img"])
                preds = self.postprocess(preds)

                # Calculate metrics
                metrics = self.validator.get_metrics(preds, batch)

                # Save all metrics together
                self.validator.save_metrics(
                    metrics=metrics.results_dict,
                    epoch=self.current_epoch,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    time_epoch=epoch_time,
                    learning_rates=lrs,
                )

        # Set model back to training mode
        self.model.train()

        # Reset epoch start time
        self.epoch_start_time = time.time()

        # Original confusion matrix and metrics plotting
        self.validator.confusion_matrix.plot(
            save_dir=self.save_dir,
            names=self.model.names.values(),
            normalize=True,
        )
        self.plot_metrics()

        # Evaluate model on validation dataset
        self.model.eval()
        class_metrics = {
            name: {"tp": 0, "fp": 0, "fn": 0} for name in self.model.names.values()
        }

        with torch.no_grad():
            for batch in self.trainer.datamodule.val_dataloader():
                # Prepare batch
                org_img = batch["img"].to(self.device).float()
                batch["img"] = org_img / 255
                for k in ["batch_idx", "cls", "bboxes"]:
                    batch[k] = batch[k].to(self.device)

                # Get predictions
                preds = self.model(batch["img"])
                results = self.predict_postprocess(
                    preds, batch["img"], org_img, batch["img_path"]
                )

                # Update metrics for each image in batch
                for result, true_cls in zip(results, batch["cls"]):
                    if not result.boxes:
                        # Count all ground truth boxes as false negatives
                        for cls_idx in true_cls:
                            class_name = self.model.names[int(cls_idx)]
                            class_metrics[class_name]["fn"] += 1
                        continue

                    # Count predictions
                    for box in result.boxes:
                        class_name = self.model.names[int(box.cls)]
                        if box.conf > 0.5:  # You can adjust this confidence threshold
                            class_metrics[class_name]["tp"] += 1

                    # Count ground truth
                    true_class_counts = {}
                    for cls_idx in true_cls:
                        class_name = self.model.names[int(cls_idx)]
                        true_class_counts[class_name] = (
                            true_class_counts.get(class_name, 0) + 1
                        )

                    # Calculate false negatives and false positives
                    for class_name, true_count in true_class_counts.items():
                        pred_count = sum(
                            1
                            for box in result.boxes
                            if self.model.names[int(box.cls)] == class_name
                            and box.conf > 0.5
                        )
                        if pred_count > true_count:
                            class_metrics[class_name]["fp"] += pred_count - true_count
                        elif pred_count < true_count:
                            class_metrics[class_name]["fn"] += true_count - pred_count

        # Calculate and print metrics for each class
        print("\nPer-class metrics:")
        print("Class            Precision    Recall    F1-Score")
        print("-" * 50)

        # Create a dictionary to store metrics
        metrics_dict = {}

        for class_name, metrics in class_metrics.items():
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Store metrics in dictionary
            metrics_dict[class_name] = {
                "precision": float(
                    precision
                ),  # Convert to float for JSON serialization
                "recall": float(recall),
                "f1": float(f1),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }

            print(f"{class_name:<15} {precision:>9.3f} {recall:>9.3f} {f1:>9.3f}")

        # Save metrics to JSON file
        import json

        metrics_path = self.save_dir / "class_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"\nSaved class metrics to {metrics_path}")

        # Save final model
        self.save_model("final.pt")
        super().on_train_end()

    def postprocess(self, preds, conf_thres=0.2, iou_thres=0.7):
        """Apply Non-maximum suppression to prediction outputs.

        Args:
            preds: Model predictions
            conf_thres (float): Confidence threshold for detections (0-1)
            iou_thres (float): NMS IoU threshold (0-1)
        """
        return ops.non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=True,
        )

    def predict_postprocess(
        self, preds, img, orig_imgs, img_paths, conf_thres=0.2, iou_thres=0.7
    ):
        """Post-processes predictions and returns a list of Results objects.

        Args:
            preds: Model predictions
            img: Input image tensor
            orig_imgs: Original images
            img_paths: Paths to input images
            conf_thres (float): Confidence threshold for detections (0-1)
            iou_thres (float): NMS IoU threshold (0-1)
        """
        preds = ops.non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, img_paths):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
            )
        return results

    def plot_metrics(self):
        """Plot training metrics from the training results CSV file."""
        plot_results(
            file=Path(self.save_dir) / "results.csv"
        )  # plot results.csv as results.png

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
            "optimizer": convert_optimizer_state_dict_to_fp16(
                deepcopy(self.trainer.optimizers[0].state_dict())
            ),
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

    def evaluate_dataloader(
        self,
        dataloader,
        save_prefix="",
        device="cuda",
        conf_thres=0.2,
        iou_thres=0.7,
    ):
        """Evaluate model on a dataloader and compute basic metrics.

        Args:
            dataloader: DataLoader to evaluate on
            save_prefix (str): Prefix to add to saved metrics file
        """
        self.model.to(device)
        # Set model to eval mode
        self.model.eval()

        # Basic metrics tracking
        class_metrics = {
            name: {"tp": 0, "fp": 0, "fn": 0} for name in self.model.names.values()
        }

        with torch.no_grad():
            # Add tqdm progress bar
            pbar = tqdm(dataloader, desc=f"Evaluating {save_prefix}", leave=True)
            for batch in pbar:
                # Prepare batch
                org_img = batch["img"].to(device).float()
                batch["img"] = org_img / 255
                for k in ["batch_idx", "cls", "bboxes"]:
                    batch[k] = batch[k].to(device)

                # Get predictions
                preds = self.model(batch["img"])
                results = self.predict_postprocess(
                    preds,
                    batch["img"],
                    org_img,
                    batch["img_path"],
                    conf_thres,
                    iou_thres,
                )

                # Update metrics for each image in batch
                for result, true_cls in zip(results, batch["cls"]):
                    if not result.boxes:
                        # Count all ground truth boxes as false negatives
                        for cls_idx in true_cls:
                            class_name = self.model.names[int(cls_idx)]
                            class_metrics[class_name]["fn"] += 1
                        continue

                    # Count predictions
                    for box in result.boxes:
                        class_name = self.model.names[int(box.cls)]
                        if box.conf > 0.5:  # Confidence threshold
                            class_metrics[class_name]["tp"] += 1

                    # Count ground truth
                    true_class_counts = {}
                    for cls_idx in true_cls:
                        class_name = self.model.names[int(cls_idx)]
                        true_class_counts[class_name] = (
                            true_class_counts.get(class_name, 0) + 1
                        )

                    # Calculate false negatives and false positives
                    for class_name, true_count in true_class_counts.items():
                        pred_count = sum(
                            1
                            for box in result.boxes
                            if self.model.names[int(box.cls)] == class_name
                            and box.conf > 0.5
                        )
                        if pred_count > true_count:
                            class_metrics[class_name]["fp"] += pred_count - true_count
                        elif pred_count < true_count:
                            class_metrics[class_name]["fn"] += true_count - pred_count

        # Calculate and store metrics
        metrics_dict = {}
        print(
            f"\n{save_prefix} Per-class metrics at conf_thres={conf_thres} and iou_thres={iou_thres}:"
        )
        print("Class            Precision    Recall    F1-Score")
        print("-" * 50)

        for class_name, metrics in class_metrics.items():
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics_dict[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
            }

            print(f"{class_name:<15} {precision:>9.3f} {recall:>9.3f} {f1:>9.3f}")

        # Save metrics to JSON file
        metrics_filename = (
            f"{save_prefix}_metrics.json" if save_prefix else "metrics.json"
        )
        metrics_path = self.save_dir / metrics_filename
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"\nSaved {save_prefix} metrics to {metrics_path}")

        # Return model to previous mode
        self.train()

        return metrics_dict

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

    def predict(self, image, conf_thres=0.25, iou_thres=0.7, device="cuda"):
        """Perform inference on a single image.

        Args:
            image (str | np.ndarray): Path to image file or numpy array
            conf_thres (float): Confidence threshold for predictions (0-1)
            iou_thres (float): NMS IoU threshold (0-1)
            device (str): Device to run inference on ('cuda' or 'cpu')

        Returns:
            Results: Object containing predictions (boxes, scores, classes)
        """
        # Ensure model is in eval mode
        self.model.eval()
        self.model.to(device)

        # Load and preprocess image
        if isinstance(image, str):
            import cv2

            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store original image for scaling coordinates later
        orig_image = image.copy()

        # Prepare image tensor
        import numpy as np

        if isinstance(image, np.ndarray):
            # Normalize and convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC to CHW
            image = image.to(device)
            image = image.unsqueeze(0)  # Add batch dimension

        # Normalize
        image = image / 255.0

        # Inference
        with torch.no_grad():
            preds = self.model(image)

            # Apply NMS
            preds = ops.non_max_suppression(
                preds,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )

            # Process predictions
            if len(preds[0]):  # If there are detections
                # Scale boxes to original image size
                preds[0][:, :4] = ops.scale_boxes(
                    image.shape[2:], preds[0][:, :4], orig_image.shape
                )

            # Create Results object
            results = Results(
                orig_img=orig_image, path=None, names=self.model.names, boxes=preds[0]
            )

        return results


if __name__ == "__main__":
    # Example usage of YOLOv11Module
    module = YOLOv11Module(
        ckpt_path="yolo11n.pt",
        save_dir="./checkpoints/",
        override_mapping={1: "box", 2: "pen"},
    )

    # Create a sample batch for testing
    sample_batch = {
        "img": torch.rand(2, 3, 640, 640),  # 2 images, 3 channels, 640x640 resolution
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0], [1]]),  # Example class labels
        "bboxes": torch.tensor(
            [[[0.1, 0.1, 0.2, 0.2]], [[0.3, 0.3, 0.4, 0.4]]]
        ),  # Example bounding boxes
    }

    # Test forward pass
    print("Testing forward pass...")
    module.model.eval()
    with torch.no_grad():
        preds = module.model(sample_batch["img"])
        print(f"Prediction shape: {preds[0].shape}")

    print("\nModel initialized and basic tests completed successfully!")
