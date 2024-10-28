from utils import logger
from config import config
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = F.leaky_relu(self.batch_norm1(self.conv1(x)), negative_slope=0.2)
        out = self.batch_norm2(self.conv2(out))

        if self.skip_connection is not None:
            identity = self.skip_connection(identity)

        out += identity  # Residual connection
        out = F.leaky_relu(out, negative_slope=0.2)

        return out


class DocumentDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            ResidualBlock(1, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(512, 1024),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            ResidualBlock(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ResidualBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 1, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        # Encoder
        logger.info(f"Input shape: {x.shape}")
        x = self.encoder[0](x)
        logger.info(f"After first ResidualBlock: {x.shape}")
        x = self.encoder[1](x)  # MaxPool2d
        x = self.encoder[2](x)
        logger.info(f"After second ResidualBlock: {x.shape}")
        x = self.encoder[3](x)  # MaxPool2d
        x = self.encoder[4](x)
        logger.info(f"After third ResidualBlock: {x.shape}")
        x = self.encoder[5](x)  # MaxPool2d
        x = self.encoder[6](x)
        logger.info(f"After fourth ResidualBlock (encoder end): {x.shape}")

        # Decoder
        x = self.decoder[0](x)
        logger.info(f"After first ConvTranspose2d: {x.shape}")
        x = self.decoder[1](x)
        logger.info(f"After first ResidualBlock in decoder: {x.shape}")
        x = self.decoder[2](x)
        logger.info(f"After second ConvTranspose2d: {x.shape}")
        x = self.decoder[3](x)
        logger.info(f"After second ResidualBlock in decoder: {x.shape}")
        x = self.decoder[4](x)
        logger.info(f"After third ConvTranspose2d: {x.shape}")
        x = self.decoder[5](x)
        logger.info(f"After third ResidualBlock in decoder: {x.shape}")
        x = self.decoder[6](x)
        logger.info(f"After final Conv2d (output): {x.shape}")

        return torch.sigmoid(x)  # Output in the range [0, 1]
