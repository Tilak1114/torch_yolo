import os
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class TrainingConfig:
    def __init__(
        self,
        lightning_module: pl.LightningModule,
        data_module: pl.LightningDataModule,
        data_config: dict,
        training_params: dict,
        wandb_config: dict,
    ):
        self.lightning_module = lightning_module
        self.data_module = data_module
        self.data_config = data_config
        self.training_params = training_params
        self.wandb_config = wandb_config

    def __dict__(self):
        return {
            "lightning_module": self.lightning_module,
            "data_module": self.data_module,
            "data_config": self.data_config,
            "training_params": self.training_params,
            "wandb_config": self.wandb_config,
        }


class Trainer:
    def __init__(self, default_trainer_config: dict = None):
        self.default_trainer_config = default_trainer_config or {
            "accelerator": "gpu",
            "enable_checkpointing": False,
        }

    def train(self, config: TrainingConfig) -> Optional[str]:
        """
        Train the model and return the wandb run ID if logging is enabled

        Args:
            config: TrainingConfig object containing:
                - lightning_module: pl.LightningModule
                - data_module: pl.LightningDataModule
                - data_config: dict
                - training_params: dict
                - wandb_config: dict
        """
        config_dict = config.__dict__()
        return self._train(
            default_trainer_config=self.default_trainer_config,
            **config_dict,
        )

    @staticmethod
    def _train(
        lightning_module: pl.LightningModule,
        data_module: pl.LightningDataModule,
        data_config: dict,
        training_params: dict,
        wandb_config: dict,
        default_trainer_config: dict,
    ) -> str:
        print(f"Starting training...")

        logger = None
        wandb_run_id = ""

        # Only setup WandB logger if logging is enabled
        if wandb_config.get("should_log"):
            wandb_logger = WandbLogger(
                project=wandb_config.get("project", "torchyolo"),
                name=wandb_config.get("run_name"),
                group=wandb_config.get("group"),
                tags=wandb_config.get("tags", []),
                log_model=False,  # Disable model saving in wandb
                reinit=True,
                save_dir=None,  # Don't save locally through wandb
            )

            # Log configs to wandb
            wandb_logger.log_hyperparams(
                {
                    **training_params,
                    **data_config,
                }
            )

            logger = wandb_logger
            wandb_run_id = wandb_logger.experiment.id

        # Initialize trainer
        trainer_config = {**default_trainer_config}
        trainer_config.update(training_params.get("trainer", {}))

        # Handle max steps/epochs
        if "max_epochs" in training_params:
            trainer_config["max_steps"] = training_params["max_epochs"] * len(
                data_module.train_dataloader()
            )
        elif "max_steps" in training_params:
            trainer_config["max_steps"] = training_params["max_steps"]
        else:
            raise ValueError(
                "Either max_epochs or max_steps must be provided in training_params"
            )

        trainer_config["val_check_interval"] = training_params.get(
            "val_check_interval", 1.0
        )

        # Set devices from training_params
        if "gpu" in training_params:
            trainer_config["devices"] = training_params["gpu"]

        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=10,
            **trainer_config,
        )

        # Train using the data module
        trainer.fit(lightning_module, datamodule=data_module)

        if wandb_config.get("should_log"):
            wandb_logger.experiment.finish()

        return wandb_run_id
