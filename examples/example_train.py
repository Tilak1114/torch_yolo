from torchyolo.trainer import Trainer, TrainingConfig
from torchyolo.yolo.yolo_v11_module import YOLOv11Module
from torchyolo.yolo.data.dataset import YOLODataModule
import yaml
import json


def create_training_config(config: dict) -> TrainingConfig:
    data_module = YOLODataModule(
        dataset_yaml=config["data_config"]["data_yaml_path"],
        batch_size=config["data_config"]["batch_size"],
    )

    # Load the yaml and get override mapping
    with open(config["data_config"]["data_yaml_path"], "r") as f:
        data_yaml = yaml.safe_load(f)
    override_mapping = data_yaml.get("names")

    # Create YOLO module
    yolov11_module = YOLOv11Module(
        ckpt_path=config["training_params"]["pretrained_model_ckpt"],
        override_mapping=override_mapping,
        save_dir=config["training_params"]["output_dir"],
        save_every=config["training_params"]["save_every"],
        freeze_backbone=config["training_params"]["freeze_backbone"],
        profiler_config=config["profiler_config"],
    )

    return TrainingConfig(
        lightning_module=yolov11_module,
        data_module=data_module,
        data_config=config["data_config"],
        training_params=config["training_params"],
        wandb_config=config["wandb_config"],
    )


if __name__ == "__main__":
    trainer = Trainer()

    with open("configs/sample_config.json", "r") as f:
        config = json.load(f)

    trainer.train(create_training_config(config))
