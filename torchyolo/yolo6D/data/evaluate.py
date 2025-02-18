from pathlib import Path
from pipelearn.models.yolo6D.yolo6d_module import YOLO6DModule
from pipelearn.models.yolo6D.data.dataset import Pose6DDataModule


def evaluate_model(
    checkpoint_path: str,
    output_dir: str,
):
    """
    Evaluate a trained YOLO6D model on a dataset.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        output_dir: Directory to save evaluation results
    """
    # Initialize model and data module
    model = YOLO6DModule(
        ckpt_path=checkpoint_path,
        save_dir=output_dir,
    )

    data_module = Pose6DDataModule()
    data_module.setup()

    # Evaluate on validation set
    val_metrics = model.evaluate_dataloader(
        data_module.val_dataloader(), save_dir="val"
    )

    return val_metrics

if __name__ == "__main__":

    checkpoint_paths = [
        "/home/tilak/projects/learning/output/yolo6d_full_ds/weights/step_118000.pt",
    ]

    for checkpoint_path in checkpoint_paths:
        data_yaml_path = "/home/tilak/projects/learning/datasets/lmo/data.yaml"
        # Run evaluation
        # Create output directory
        output_dir = Path(f"./evaluation_results/{checkpoint_path.split('/')[-3]}")
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = evaluate_model(
            checkpoint_path=checkpoint_path,
            output_dir=str(output_dir),
        )

        # Print summary
        print("\nEvaluation complete! Results saved to:", output_dir)
