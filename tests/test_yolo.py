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


def test_predict(yolo_v11_module):
    # Create a sample image (3 channels, 640x640)
    sample_image = torch.rand(3, 640, 640).numpy()
    sample_image = sample_image.transpose(1, 2, 0)  # CHW to HWC for numpy array
    
    # Test prediction
    results = yolo_v11_module.predict(
        sample_image,
        conf_thres=0.25,
        iou_thres=0.7,
        device='cpu'  # Use CPU for testing
    )
    
    # Verify results object structure
    assert hasattr(results, 'boxes'), "Results should have boxes attribute"
    assert hasattr(results, 'orig_img'), "Results should have orig_img attribute"
    assert hasattr(results, 'names'), "Results should have names attribute"
    
    # Verify image dimensions are preserved
    assert results.orig_img.shape == (640, 640, 3), "Original image dimensions should be preserved"
    
    # Verify names mapping
    assert results.names == {1: "box", 2: "pen"}, "Names mapping should match override_mapping"


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
