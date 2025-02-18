import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from ultralytics.data.augment import Compose, LetterBox, Format, Instances
import os
import cv2
import numpy as np
import json
from datasets import load_dataset
from embdata.geometry import Transform3D


# saving the mapping here as backup
category_id_mapping = {
    "0": {
        "diameter": 102.099,
        "min_x": -37.9343,
        "min_y": -38.7996,
        "min_z": -45.8845,
        "size_x": 75.8686,
        "size_y": 77.5992,
        "size_z": 91.769,
        "file_name": "obj_000001.ply",
        "category_name": "ape",
        "category_id": 0,
    },
    "1": {
        "diameter": 201.404,
        "min_x": -50.3958,
        "min_y": -90.8979,
        "min_z": -96.867,
        "size_x": 100.792,
        "size_y": 181.796,
        "size_z": 193.734,
        "file_name": "obj_000005.ply",
        "category_name": "can",
        "category_id": 1,
    },
    "2": {
        "diameter": 154.546,
        "min_x": -33.5054,
        "min_y": -63.8165,
        "min_z": -58.7283,
        "size_x": 67.0107,
        "size_y": 127.633,
        "size_z": 117.457,
        "file_name": "obj_000006.ply",
        "category_name": "cat",
        "category_id": 2,
    },
    "3": {
        "diameter": 261.472,
        "min_x": -114.738,
        "min_y": -37.7357,
        "min_z": -104.001,
        "size_x": 229.476,
        "size_y": 75.4714,
        "size_z": 208.002,
        "file_name": "obj_000008.ply",
        "category_name": "driller",
        "category_id": 3,
    },
    "4": {
        "diameter": 108.999,
        "min_x": -52.2146,
        "min_y": -38.7038,
        "min_z": -42.8485,
        "size_x": 104.429,
        "size_y": 77.4076,
        "size_z": 85.697,
        "file_name": "obj_000009.ply",
        "category_name": "duck",
        "category_id": 4,
    },
    "5": {
        "diameter": 164.628,
        "min_x": -75.0923,
        "min_y": -53.5375,
        "min_z": -34.6207,
        "size_x": 150.185,
        "size_y": 107.075,
        "size_z": 69.2414,
        "file_name": "obj_000010.ply",
        "category_name": "eggbox",
        "category_id": 5,
        "symmetries_discrete": [
            [
                -0.999964,
                -0.00333777,
                -0.0077452,
                0.232611,
                0.00321462,
                -0.999869,
                0.0158593,
                0.694388,
                -0.00779712,
                0.0158338,
                0.999844,
                -0.0792063,
                0,
                0,
                0,
                1,
            ]
        ],
    },
    "6": {
        "diameter": 175.889,
        "min_x": -18.3605,
        "min_y": -38.933,
        "min_z": -86.4079,
        "size_x": 36.7211,
        "size_y": 77.866,
        "size_z": 172.816,
        "file_name": "obj_000011.ply",
        "category_name": "glue",
        "category_id": 6,
        "symmetries_discrete": [
            [
                -0.999633,
                0.026679,
                0.00479336,
                -0.262139,
                -0.0266744,
                -0.999644,
                0.00100504,
                -0.197966,
                0.00481847,
                0.000876815,
                0.999988,
                0.0321652,
                0,
                0,
                0,
                1,
            ]
        ],
    },
    "7": {
        "diameter": 145.543,
        "min_x": -50.4439,
        "min_y": -54.2485,
        "min_z": -45.4,
        "size_x": 100.888,
        "size_y": 108.497,
        "size_z": 90.8,
        "file_name": "obj_000012.ply",
        "category_name": "holepuncher",
        "category_id": 7,
    },
}


class Pose6DDataset(Dataset):
    def __init__(
        self,
        dataset_name="mbodiai/bop-lmo-episode",
        split="train",
        imgsz=(480, 640),
    ):
        self.dataset = load_dataset(dataset_name, split=split)
        self.imgsz = imgsz
        self.transform = Compose(
            [
                torch.tensor,
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = np.array(data["observation"]["image"]["bytes"])
        camera = data["observation"]["camera"]

        # intrinsics = np.array(list(camera["intrinsics"].values()))

        objects = data["action"]["objects"]

        labels = []
        bboxes = []
        rotations = []
        cx_cys = []
        tzs = []

        for obj in objects:
            pose = obj["pose"]
            transform3d = Transform3D.from_pose(np.array(list(pose.values())))
            rotation = transform3d.rotation
            translation = transform3d.translation

            tz = translation[2]

            class_id = int(obj["name"])
            bbox = obj["bbox_2d"]
            x1, y1, x2, y2 = map(int, bbox.values())

            # convert to cx, cy, w, h
            cx = round((x1 + x2) / 2)
            cy = round((y1 + y2) / 2)
            w = round(x2 - x1)
            h = round(y2 - y1)

            # normalize cx, cy, w, h
            cx = cx / image.shape[1]
            cy = cy / image.shape[0]
            w = w / image.shape[1]
            h = h / image.shape[0]

            labels.append(class_id)
            bboxes.append([cx, cy, w, h])
            rotations.append(rotation)
            tzs.append(tz)
            cx_cys.append([cx, cy])

        # Convert lists to tensors
        image = self.transform(image)
        image = image.permute(2, 0, 1)  # Change from HWC to CHW format
        labels = torch.tensor(labels).unsqueeze(1)
        bboxes = torch.tensor(bboxes)
        rotations = torch.tensor(rotations)
        tzs = torch.tensor(tzs).unsqueeze(1)
        cx_cys = torch.tensor(cx_cys)

        return {
            "image": image,
            "camera": camera,
            "labels": labels,
            "bboxes": bboxes,
            "rotations": rotations,
            "tzs": tzs,
            "cx_cys": cx_cys,
        }

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches for Pose6DDataset.

        Args:
            batch (list): List of dictionaries with keys:
                - "image": Tensor of shape (C, H, W)
                - "labels": Tensor of shape (N,)
                - "bboxes": Tensor of shape (N, 4)
                - "rotations": Tensor of shape (N, 9)
                - "del_zs": Tensor of shape (N,)

        Returns:
            dict: Collated batch with concatenated object-level data and stacked images.
        """
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            value = values[i]
            if k == "image":
                # Stack all images along the batch dimension
                value = torch.stack(value, 0)
            elif k in {"labels", "bboxes", "rotations", "tzs", "cx_cys"}:
                # Concatenate object-level data
                value = torch.cat(value, 0)
            else:
                # Handle any other keys directly
                value = value
            new_batch[k] = value

        # Build `batch_idx` to track image index for each object
        new_batch["batch_idx"] = torch.cat(
            [
                torch.full((len(b["labels"]),), i, dtype=torch.int64)
                for i, b in enumerate(batch)
            ]
        )

        return new_batch


class Pose6DDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, imgsz=(480, 640), val_split: float = 0.1):
        super().__init__()
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.setup()

    def setup(self, stage: str = None):
        # Create full train dataset
        full_train_dataset = Pose6DDataset(split="train", imgsz=self.imgsz)

        # Split the train dataset into train and validation
        train_size = int((1 - self.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )
        self.test_dataset = Pose6DDataset(
            dataset_name="mbodiai/bop-lm-episode", split="test", imgsz=self.imgsz
        )

    def train_dataloader(self):
        return (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=Pose6DDataset.collate_fn,
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
                collate_fn=Pose6DDataset.collate_fn,
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
                collate_fn=Pose6DDataset.collate_fn,
            )
            if self.test_dataset
            else None
        )


def main():
    data_module = Pose6DDataModule(batch_size=16)
    train_loader = data_module.train_dataloader()

    for batch in train_loader:
        print(batch)
        break  # Iterate over only one batch


if __name__ == "__main__":
    main()
