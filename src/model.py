import torch
import torch.nn as nn
import torchvision.transforms.functional as FT
import torch.nn.functional as F

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = FT.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.down_conv = nn.ModuleList([
            DoubleConvolution(i, o) for i, o in [
                (in_channels, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, 1024),
                (1024, 2048),
            ]
        ])

        self.down_sample = nn.ModuleList([DownSample() for _ in range(5)])

        self.middle_conv = DoubleConvolution(1024, 2048)

        self.up_sample = nn.ModuleList([
            UpSample(i, o) for i, o in [
                (2048, 1024),
                (1024, 512),
                (512, 256),
                (256, 128),
                (128, 64)
            ]
        ])

        self.up_conv = nn.ModuleList([
            DoubleConvolution(i, o) for i, o in [
                (2048, 1024),
                (1024, 512),
                (512, 256),
                (256, 128),
                (128, 64)
            ]
        ])

        self.crop_and_concat = nn.ModuleList([CropAndConcat() for _ in range(5)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        pass_through = []

        for i in range(5):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(5):
            x = self.up_sample[i](x)
            x = self.crop_and_concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)

        return x


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = UNet(
            in_channels=3,
            out_channels=3,
        )

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, x):
        out = self.model(x)

        w, h = config.image.image_shape
        out = F.interpolate(out, size=(w, h), mode='bilinear', align_corners=False)

        return out
