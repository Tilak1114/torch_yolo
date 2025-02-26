from torchyolo.yolo.yolo_v11_module import YOLOv11Module
from torchyolo.yolo.yolo11_model import DEFAULT_NAME_MAPPING
import cv2
import numpy as np
from pathlib import Path

def run_inference(
    image_path: str,
    model_path: str,
    save_dir: str = "inference_output",
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    device: str = "cuda",
):
    """
    Run inference on a single image using YOLOv11Module.

    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the model checkpoint
        save_dir (str): Directory to save results
        conf_thres (float): Confidence threshold for detections
        iou_thres (float): NMS IoU threshold
        device (str): Device to run inference on ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing detection results
    """
    # Initialize model
    model = YOLOv11Module(
        ckpt_path=model_path,
        save_dir=save_dir,
        override_mapping=DEFAULT_NAME_MAPPING,
    )

    # Run prediction
    results = model.predict(
        image_path,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device
    )

    # Get the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare results dictionary
    detection_results = {
        "detections": [],
        "image": image  # Add original image to results
    }

    if results.boxes is not None:
        for box in results.boxes:
            detection_results["detections"].append({
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "confidence": float(box.conf),
                "class": model.model.names[int(box.cls)],
            })

    return detection_results

if __name__ == "__main__":
    # Example usage
    image_path = "sample_data/images/train/2083.jpg"
    model_path = "yolo11x.pt"
    save_dir = "inference_output"
    
    print(f"Running inference on {image_path}")
    print(f"Using model: {model_path}")
    
    results = run_inference(
        image_path=image_path,
        model_path=model_path,
        save_dir=save_dir,
        conf_thres=0.25,
        iou_thres=0.7,
    )

    # Get the image for visualization
    image = results["image"].copy()

    # Draw boxes on the image
    for detection in results["detections"]:
        # Get box coordinates and info
        x1, y1, x2, y2 = map(int, detection["bbox"])
        conf = detection["confidence"]
        class_name = detection["class"]

        # Define colors for different classes
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # Draw rectangle with thicker lines
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Improve label visibility
        label = f"{class_name} {conf:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0], y1),
            color,
            -1,  # Fill rectangle
        )
        
        # Draw label text in white
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2,
        )

    # Create output directory and save the annotated image
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"inference_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Print results
    print("\nDetection Results:")
    for i, detection in enumerate(results["detections"], 1):
        print(f"\nDetection {i}:")
        print(f"Class: {detection['class']}")
        print(f"Confidence: {detection['confidence']:.2f}")
        print(f"Bounding Box: {detection['bbox']}")
    
    print(f"\nAnnotated image saved as: {output_path}")