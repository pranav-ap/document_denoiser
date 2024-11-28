import os

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from neptune.types import File

from config import config
from src import Denoiser
from utils import create_image_column, create_three_image_row, get_tensor_grid

torch.set_float32_matmul_precision('medium')


class Light(pl.LightningModule):
    def __init__(self, neptune_logger=None, tensorboard_logger=None):
        super().__init__()

        self.neptune_logger = neptune_logger
        self.tensorboard_logger = tensorboard_logger

        self.model = Denoiser()
        self.learning_rate = config.train.learning_rate

        self.save_hyperparameters({
            'learning_rate': self.learning_rate,
        },
            ignore=[
                'model',
                'neptune_logger',
                'tensorboard_logger'
            ]
        )

    def forward(self, noisy_images):
        return self.model(noisy_images)

    @staticmethod
    def compute_loss(restored_images, clean_images):
        return F.mse_loss(restored_images, clean_images)

    def training_step(self, batch, batch_idx):
        noisy_images, clean_images = batch
        restored_images = self(noisy_images)
        
        loss = self.compute_loss(restored_images, clean_images)
        self.log(f"train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.save_count
            self._log_images(
                noisy_images, restored_images, clean_images,
                limit_count=limit_count,
                stage='train'
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        noisy_images, clean_images = batch
        restored_images = self(noisy_images)

        loss = self.compute_loss(restored_images, clean_images)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.save_count
            self._log_images(
                noisy_images, restored_images, clean_images,
                limit_count=limit_count,
                stage='validate'
            )

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        noisy_images, clean_images = batch
        restored_images = self(noisy_images)

        loss = self.compute_loss(restored_images, clean_images)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.save_count
            self._log_images(
                noisy_images, restored_images, clean_images,
                limit_count=limit_count,
                stage='test'
            )

        return loss

    def _log_images(self, noisy_images, restored_images, clean_images, limit_count=None, stage=None):
        rows = []

        for i, (noisy_image, restored_image, clean_image) in enumerate(zip(noisy_images, restored_images, clean_images)):
            row = create_three_image_row(noisy_image, restored_image, clean_image)
            rows.append(row)

            if i == limit_count:
                break

        final = create_image_column(rows)

        out_path = config.paths.output.test_images

        if stage == 'train':
            out_path = config.paths.output.train_images
        elif stage == 'validate':
            out_path = config.paths.output.val_images

        name = f'{stage}_epoch_{self.current_epoch}.png'
        out_path = os.path.join(out_path, name)

        final.save(out_path)

        if self.neptune_logger is not None:
            self.neptune_logger.experiment[f"{stage}/images"].append(
                File.as_image(final),
                step=self.global_step,
                name=name,
            )

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.experiment.add_images(
                tag=f"{stage}_images",
                img_tensor=get_tensor_grid(final),
                global_step=self.global_step
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=2,
                factor=0.5
            ),
            "monitor": "train/loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.train.patience,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            dirpath=config.paths.output.checkpoints,
            filename="best_checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=5)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            lr_monitor_callback
        ]
