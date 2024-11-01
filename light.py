from utils import logger
from config import config
import os
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from utils import create_three_image_row


torch.set_float32_matmul_precision('medium')


class DocumentDenoiserLightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.train.learning_rate
        self.save_hyperparameters(ignore=['model'])

        os.makedirs(config.dirs.test_images, exist_ok=True)
        os.makedirs(config.dirs.val_images, exist_ok=True)

    def forward(self, noisy_images):
        return self.model(noisy_images)

    @staticmethod
    def compute_loss(restored_images, clean_images):
        return F.mse_loss(restored_images, clean_images)

    def training_step(self, batch, batch_idx):
        noisy_images, clean_images = batch
        restored_images = self(noisy_images)
        
        loss = self.compute_loss(restored_images, clean_images)
        self.log(f"train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        noisy_images, clean_images = batch
        restored_images = self(noisy_images)

        loss = self.compute_loss(restored_images, clean_images)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0 or batch_idx == 1:
            for i, (noisy_image, restored_image, clean_image) in enumerate(zip(noisy_images, restored_images, clean_images)):
                combined_image = create_three_image_row(noisy_image, restored_image, clean_image)
                save_path = os.path.join(config.dirs.val_images, f"val_image_{batch_idx}_{i}.png")
                combined_image.save(save_path)

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        noisy_images, clean_images = batch
        restored_images = self(noisy_images)

        loss = self.compute_loss(restored_images, clean_images)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0 or batch_idx == 1:
            for i, (noisy_image, restored_image, clean_image) in enumerate(zip(noisy_images, restored_images, clean_images)):
                combined_image = create_three_image_row(noisy_image, restored_image, clean_image)
                save_path = os.path.join(config.dirs.test_images, f"test_image_{batch_idx}_{i}.png")
                combined_image.save(save_path)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5),
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    # def configure_optimizers(self):
    #     from torch.optim.lr_scheduler import CosineAnnealingLR
    #     optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
    #     lr_scheduler = {
    #         "scheduler": CosineAnnealingLR(optimizer, T_max=config.train.max_epochs, eta_min=1e-6),  
    #         "interval": "epoch",
    #         "frequency": 1,
    #     }
        
    #     return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        
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
