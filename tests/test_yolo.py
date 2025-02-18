import pytest
import torch

from torchyolo.yolo.yolo_v11_module import YOLOv11Module


@pytest.fixture
def yolo_v11_module():
    ckpt_path = "yolo11n.pt"
    save_dir = "./checkpoints/"
    return YOLOv11Module(
        ckpt_path=ckpt_path, override_mapping={1: "box", 2: "pen"}, save_dir=save_dir
    )


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
