from utils.logger_setup import logger
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
import lightning.pytorch as pl

torch.set_float32_matmul_precision('medium')


"""
Configuration
"""


@dataclass
class TrainingConfig:
    train_batch_size = 2
    val_batch_size = 2

    max_epochs = 20
    check_val_every_n_epoch = 4
    log_every_n_steps = 25
    accumulate_grad_batches = 32
    learning_rate = 1e-4

    data_dir = "D:/document_denoiser/data/augraphy"
    output_dir = "D:/document_denoiser/data/augraphy"
    # data_dir = "/kaggle/input"
    # output_dir = "/kaggle/working"


config = TrainingConfig()


"""
Dataset Classes
"""


class DocDenoiserDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, transform, stage):
        super().__init__()
        self.filenames = filenames
        self.transform = transform
        self.stage = stage

    def __len__(self):
        length = len(self.filenames)
        return length

    def get_test_image(self, index):
        filename = self.filenames[index]

        noisy_image = torchvision.io.read_image(
            os.path.join(f'{config.data_dir}/test/test_shabby', filename)
        )

        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image.float()

    def __getitem__(self, index):
        if self.stage == 'test':
            return self.get_test_image(index)

        filename = self.filenames[index]

        noisy_image = torchvision.io.read_image(
            os.path.join(f'{config.data_dir}/train/train_shabby', filename)
        )

        if self.transform:
            noisy_image = self.transform(noisy_image)

        clean_image = torchvision.io.read_image(
            os.path.join(f'{config.data_dir}/train/train_cleaned', filename)
        )

        if self.transform:
            clean_image = self.transform(clean_image)

        return noisy_image.float(), clean_image.float()


class DocDenoiserDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()  # <- use all available CPU cores
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            self.num_workers = 2 * num_gpus

        from torchvision import transforms as T

        self.transform = T.Compose([
            # T.Resize((420, 540)),
            # T.Grayscale(num_output_channels=1),
            # Convert image to tensor (scales pixel values to [0, 1])
            # T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            # Normalize to have mean 0.5 and std 0.5
            T.Normalize((0.5,), (0.5,)),
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            all_filenames = [
                filename
                for filename in os.listdir(f'{config.data_dir}/train/train_cleaned')
                if os.path.isfile(os.path.join(f'{config.data_dir}/train/train_shabby', filename)) and
                os.path.isfile(os.path.join(f'{config.data_dir}/train/train_cleaned', filename))
            ]

            from sklearn.model_selection import train_test_split
            train_filenames, val_filenames = train_test_split(
                all_filenames,
                test_size=0.2,
                shuffle=True
            )

            self.train_dataset = DocDenoiserDataset(
                filenames=train_filenames,
                transform=self.transform,
                stage='train'
            )

            self.val_dataset = DocDenoiserDataset(
                filenames=val_filenames,
                transform=self.transform,
                stage='validate'
            )

            logger.info(f"Total Dataset       : {len(self.train_dataset) + len(self.val_dataset)} samples")
            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == 'test':
            test_filenames = [
                filename
                for filename in os.listdir(f'{config.data_dir}/test/test_cleaned')
                if os.path.isfile(os.path.join(f'{config.data_dir}/test/test_cleaned', filename))
            ]

            self.test_dataset = DocDenoiserDataset(
                filenames=test_filenames,
                transform=self.transform,
                stage='test'
            )

            logger.info(f"Test Dataset  : {len(self.test_dataset)} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


"""
Model Classes
"""


class DocDenoiserAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder layers

        self.enc_1 = nn.Conv2d(1, 128, kernel_size=3, padding='same')
        self.enc_2 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.enc_bat_1 = nn.BatchNorm2d(256)
        self.enc_maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_3 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.enc_4 = nn.Conv2d(512, 1024, kernel_size=3, padding='same')
        self.enc_bat_2 = nn.BatchNorm2d(1024)
        self.enc_maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(0.5)

        # decoder layers

        self.dec_0 = nn.Conv2d(1024, 512, kernel_size=3, padding='same')
        self.dec_bat_1 = nn.BatchNorm2d(512)
        self.dec_upsample_1 = nn.Upsample(scale_factor=2)

        self.dec_2 = nn.Conv2d(512, 256, kernel_size=3, padding='same')
        self.dec_3 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.dec_4 = nn.Conv2d(128, 1, kernel_size=3, padding='same')
        self.dec_upsample_2 = nn.Upsample(scale_factor=2)

        self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        # Encoder

        x = F.relu(self.enc_1(x))
        x = F.relu(self.enc_2(x))
        x = self.enc_bat_1(x)
        x = self.enc_maxpool_1(x)

        x = F.relu(self.enc_3(x))
        x = F.relu(self.enc_4(x))
        x = self.enc_bat_2(x)
        x = self.enc_maxpool_2(x)

#         x = self.drop1(x)

        # Decoder

        x = F.relu(self.dec_0(x))
#         x = self.dec_bat_1(x)
        x = self.dec_upsample_1(x)

        x = F.relu(self.dec_2(x))
        x = self.dec_upsample_2(x)

        x = F.relu(self.dec_3(x))
        x = F.relu(self.dec_4(x))

#         x = self.drop2(x)

        x = F.sigmoid(x)

        return x


"""
Lightning Module
"""


class DocDenoiserLightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.learning_rate

        # self.save_hyperparameters(ignore=['model'])

    def forward(self, noisy_images):
        restored_images = self.model(noisy_images)
        return restored_images

    def shared_step(self, batch):
        noisy_images, clean_images = batch
        restored_images = self.model(noisy_images)
        loss = F.mse_loss(restored_images, clean_images)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            }
        }

    def configure_callbacks(self):
        early_stop = L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.00,
            patience=4,
            verbose=False,
        )

        checkpoint = L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.output_dir}/checkpoints/',
            save_top_k=1,
            save_last=True
        )

        progress_bar = L.pytorch.callbacks.TQDMProgressBar(process_position=0)
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        # summary = ModelSummary(max_depth=-1)
        # swa = StochasticWeightAveraging(swa_lrs=1e-2)

        return [checkpoint, progress_bar, lr_monitor, early_stop]


"""
Train Function
"""


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    autoencoder_model = DocDenoiserAutoencoder()
    lightning_module = DocDenoiserLightning(autoencoder_model)
    dm = DocDenoiserDataModule()

    trainer = pl.Trainer(
        default_root_dir=f"{config.output_dir}/",
        logger=L.pytorch.loggers.CSVLogger(save_dir=f'{config.output_dir}/'),
        devices='auto',
        accelerator="auto",  # auto, gpu, cpu, ...

        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        # gradient_clip_val=0.1,

        fast_dev_run=True,
        # overfit_batches=1,
        num_sanity_val_steps=1,
        enable_model_summary=False,
    )

    trainer.fit(
        lightning_module,
        datamodule=dm,
        # ckpt_path=f'{config.output_dir}\\checkpoints\\last.ckpt'
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best model path : {best_model_path}")



