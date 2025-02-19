import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from ultralytics.data.augment import Compose, LetterBox, Format, Instances
import os
import cv2
import numpy as np
import yaml


class YOLODataset(Dataset):
    def __init__(
        self, img_dir, labels_dir, imgsz=640, mask_ratio=0.4, overlap_mask=0.0, bgr=0.0
    ):
        self.img_path = img_dir
        self.labels_path = labels_dir
        self.imgsz = imgsz

        # Filter to only keep images that have corresponding label files
        all_im_files = os.listdir(img_dir)
        self.label_files = set(os.listdir(labels_dir))
        self.im_files = [
            f
            for f in all_im_files
            if os.path.splitext(f)[0] + ".txt" in self.label_files
        ]

        print(
            f"Found {len(self.im_files)} valid image-label pairs out of {len(all_im_files)} images"
        )
        self.transforms = self.get_transforms(mask_ratio, overlap_mask, bgr)

    def get_transforms(self, mask_ratio=0.4, overlap_mask=0.0, bgr=0.0):
        transforms = Compose(
            [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
        )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=mask_ratio,
                mask_overlap=overlap_mask,
                bgr=bgr,  # only affect training.
            )
        )
        return transforms

    def __len__(self):
        return len(self.im_files)

    def get_label(self, label_file):
        label_path = os.path.join(self.labels_path, label_file)
        cls_list = []
        bbox_list = []

        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                cls = int(values[0])
                bbox = [float(x) for x in values[1:5]]

                cls_list.append([cls])
                bbox_list.append(bbox)

        return {
            "cls": np.array(cls_list),
            "bboxes": np.array(bbox_list),
            "bbox_format": "xywh",
            "normalized": True,
        }

    def get_image_and_label(self, index):
        img_file = self.im_files[index]
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label = self.get_label(label_file)
        img_file_path = os.path.join(self.img_path, img_file)

        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(
            img_file_path
        )
        label["img_path"] = img_file_path
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation

        return self.update_labels_info(label)

    def load_image(self, img_file):
        im = cv2.imread(img_file)  # BGR
        if im is None:
            raise FileNotFoundError(f"Image Not Found {img_file}")

        h0, w0 = im.shape[:2]  # orig hw
        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        return im, (h0, w0), im.shape[:2]

    def update_labels_info(self, label):
        bboxes = label.pop("bboxes")
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        segment_resamples = 1000
        segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)

        label["instances"] = Instances(
            bboxes,
            segments=segments,
            keypoints=None,
            bbox_format=bbox_format,
            normalized=normalized,
        )
        return label

    def __getitem__(self, idx):
        return self.transforms(self.get_image_and_label(idx))

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLODataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_yaml: str,
        batch_size: int = 16,
        imgsz: int = 640,
        num_workers: int = 4,
    ):
        super().__init__()
        # Read YAML file
        with open(dataset_yaml, "r") as f:
            self.yaml_data = yaml.safe_load(f)

        self.data_path = self.yaml_data["path"]
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.setup()

    def setup(self, stage: str = None):
        # Create train dataset
        train_img_dir = os.path.join(self.data_path, self.yaml_data["train"])
        train_labels_dir = train_img_dir.replace("images", "labels")
        self.train_dataset = YOLODataset(
            img_dir=train_img_dir, labels_dir=train_labels_dir, imgsz=self.imgsz
        )

        # Create validation dataset if specified
        if "val" in self.yaml_data:
            val_img_dir = os.path.join(self.data_path, self.yaml_data["val"])
            val_labels_dir = val_img_dir.replace("images", "labels")
            self.val_dataset = YOLODataset(
                img_dir=val_img_dir, labels_dir=val_labels_dir, imgsz=self.imgsz
            )

        # Create test dataset if specified
        if "test" in self.yaml_data:
            test_img_dir = os.path.join(self.data_path, self.yaml_data["test"])
            test_labels_dir = test_img_dir.replace("images", "labels")
            self.test_dataset = YOLODataset(
                img_dir=test_img_dir,
                labels_dir=test_labels_dir,
                imgsz=self.imgsz,
            )

    def train_dataloader(self):
        return (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=YOLODataset.collate_fn,
                num_workers=self.num_workers,
            )
            if self.train_dataset
            else None
        )

    def val_dataloader(self):
        return (
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=YOLODataset.collate_fn,
                num_workers=self.num_workers,
            )
            if self.val_dataset
            else None
        )

    def test_dataloader(self):
        return (
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=YOLODataset.collate_fn,
                num_workers=self.num_workers,
            )
            if self.test_dataset
            else None
        )
