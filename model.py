from utils import logger
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
            ResidualBlock(1, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ResidualBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=3, padding='same'),
        )
    
    def forward(self, x_in):
        # Encoder
        logger.debug(f"Input shape: {x_in.shape}")
        x = self.encoder[0](x_in)
        logger.debug(f"After first ResidualBlock: {x.shape}")
        x = self.encoder[1](x)  # MaxPool2d
        x = self.encoder[2](x)
        logger.debug(f"After second ResidualBlock: {x.shape}")
        x = self.encoder[3](x)  # MaxPool2d
        x = self.encoder[4](x)
        logger.debug(f"After third ResidualBlock: {x.shape}")
        x = self.encoder[5](x)  # MaxPool2d
        x = self.encoder[6](x)
        logger.debug(f"After fourth ResidualBlock (encoder end): {x.shape}")
    
        # Decoder
        x = self.decoder[0](x)
        logger.debug(f"After first ConvTranspose2d: {x.shape}")
        x = self.decoder[1](x)
        logger.debug(f"After first ResidualBlock in decoder: {x.shape}")
        x = self.decoder[2](x)
        logger.debug(f"After second ConvTranspose2d: {x.shape}")
        x = self.decoder[3](x)
        logger.debug(f"After second ResidualBlock in decoder: {x.shape}")
        x = self.decoder[4](x)
        logger.debug(f"After third ConvTranspose2d: {x.shape}")
        x = self.decoder[5](x)
        logger.debug(f"After third ResidualBlock in decoder: {x.shape}")
        x = self.decoder[6](x)
        logger.debug(f"After final Conv2d (output): {x.shape}")
    
        # Calculate desired output size based on input shape
        target_size = (x_in.size(2), x_in.size(3))  # (height, width)
        logger.debug(f"Target size for interpolation: {target_size}")
    
        # Upsample to the desired dimensions
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        logger.debug(f"After resizing to target size: {x.shape}")
    
        return torch.sigmoid(x)  # Output in the range [0, 1]

