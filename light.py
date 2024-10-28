from utils import logger
from config import config
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor

torch.set_float32_matmul_precision('medium')


class DocumentDenoiserLightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.train.learning_rate
        self.save_hyperparameters(ignore=['model'])

    def forward(self, noisy_images):
        return self.model(noisy_images)

    @staticmethod
    def compute_loss(restored_images, clean_images):
        return F.mse_loss(restored_images, clean_images)

    def shared_step(self, batch, stage: str):
        noisy_images, clean_images = batch
        restored_images = self(noisy_images)
        loss = self.compute_loss(restored_images, clean_images)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.dirs.output}/checkpoints/',
            filename="best-checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=10)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [checkpoint_callback, early_stop_callback, progress_bar_callback, lr_monitor_callback]

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss"]
        logger.info(f"Epoch [{self.current_epoch}] Training Loss: {train_loss:.4f}")

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"]
        logger.info(f"Epoch [{self.current_epoch}] Validation Loss: {val_loss:.4f}")
