from utils import logger
from config import config
import os
import torch
import lightning as L
from PIL import Image
from typing import Optional
from torchvision import transforms as T


torch.set_float32_matmul_precision('medium')


class AugraphyDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self, root_dir, subset, transform=None, sample_size=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

        self.shabby_dir = os.path.join(root_dir, subset, subset, f'{subset}_shabby')
        self.clean_dir = os.path.join(root_dir, subset, subset, f'{subset}_cleaned')

        self.shabby_images = sorted(os.listdir(self.shabby_dir))
        self.clean_images = sorted(os.listdir(self.clean_dir))

        if sample_size is not None:
            sample_indices = random.sample(range(len(self.shabby_images)), min(sample_size, len(self.shabby_images)))
            self.shabby_images = [self.shabby_images[i] for i in sample_indices]
            self.clean_images = [self.clean_images[i] for i in sample_indices]

        assert len(self.shabby_images) == len(self.clean_images), "Mismatch in number of shabby and clean images."
        
    def __len__(self):
        return len(self.shabby_images)

    def __getitem__(self, idx):
        shabby_path = os.path.join(self.shabby_dir, self.shabby_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])

        shabby_image = Image.open(shabby_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        if self.transform:
            shabby_image = self.transform(shabby_image)
            clean_image = self.transform(clean_image)

        return shabby_image, clean_image


class DocumentDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()

        w, h = config.image_size

        self.transform = T.Compose([
            T.Resize((w, h)),
            T.Grayscale(),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.train_dataset: Optional[AugraphyDataset] = None
        self.val_dataset: Optional[AugraphyDataset] = None
        self.test_dataset: Optional[AugraphyDataset] = None

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            root_dir = config.dirs.data
            self.train_dataset = AugraphyDataset(root_dir=root_dir, subset="train", transform=self.transform)
            self.val_dataset = AugraphyDataset(root_dir=root_dir, subset="validate", transform=self.transform)

            logger.info(f"Total Dataset       : {len(self.train_dataset) + len(self.val_dataset)} samples")
            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == "test":
            root_dir = config.dirs.data
            self.test_dataset = AugraphyDataset(root_dir=root_dir, subset="test", transform=self.transform)
            logger.info(f"Test Dataset  : {len(self.test_dataset)} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.train.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.test.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
