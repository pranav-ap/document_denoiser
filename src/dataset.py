import os
from typing import Dict, Optional

import albumentations as A
import lightning as L
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class DenoiserDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self, subset, transform=None, augmentation=None):
        self.subset = subset
        self.transform = transform
        self.augmentation = augmentation

        root_dir = config.paths.roots.data
        self.shabby_dir: str = os.path.join(root_dir, subset, subset, f'{subset}_shabby')
        self.clean_dir: str = os.path.join(root_dir, subset, subset, f'{subset}_cleaned')

        self.shabby_images = sorted(os.listdir(self.shabby_dir))
        self.clean_images = sorted(os.listdir(self.clean_dir))

        assert len(self.shabby_images) == len(self.clean_images), "Mismatch in number of shabby and clean images."
        
    def __len__(self):
        return len(self.shabby_images)

    def __getitem__(self, idx):
        shabby_path = os.path.join(self.shabby_dir, self.shabby_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])

        shabby_image = Image.open(shabby_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        if self.augmentation:
            shabby_image = self.augmentation(image=np.array(shabby_image))['image']
            clean_image = self.augmentation(image=np.array(clean_image))['image']

            shabby_image = Image.fromarray(shabby_image)
            clean_image = Image.fromarray(clean_image)

        if self.transform:
            shabby_image = self.transform(shabby_image)
            clean_image = self.transform(clean_image)

        return shabby_image, clean_image


class DenoiserDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = 0 if config.task.eda_mode else os.cpu_count()
        self.persistent_workers = not config.task.eda_mode

        self.augmentation = A.Compose(
            transforms=[
                A.Defocus(p=0.5, radius=1),
                A.GaussNoise(p=0.5, var_limit=(10.0, 60.0)),
            ]
        )

        w, h = config.image.image_shape

        self.transform = T.Compose([
            T.Resize((w, h)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.dataset: Dict[str, Optional[DenoiserDataset]] = {}

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset['train'] = DenoiserDataset(
                subset="train",
                # augmentation=self.augmentation,
                transform=self.transform
            )

            self.dataset['val'] = DenoiserDataset(
                subset="validate",
                transform=self.transform
            )

            logger.info(f"Train Dataset       : {len(self.dataset['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.dataset['val'])} samples")

        if stage == "test":
            self.dataset['test'] = DenoiserDataset(
                subset="test",
                transform=self.transform
            )

            logger.info(f"Test Dataset  : {len(self.dataset['test'])} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['train'],
            batch_size=config.train.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['val'],
            batch_size=config.train.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['test'],
            batch_size=config.test.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )
