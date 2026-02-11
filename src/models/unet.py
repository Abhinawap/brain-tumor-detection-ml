"""
U-Net architecture for brain tumor segmentation.

This module implements the U-Net architecture with:
- Encoder: 4 downsampling blocks with max pooling
- Bottleneck: Central convolution block
- Decoder: 4 upsampling blocks with skip connections
- Output: Sigmoid activation for binary segmentation

Architecture:
    Input (3, 128, 128) → Encoder → Bottleneck → Decoder → Output (1, 128, 128)

Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Ronneberger et al., 2015

"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """
    Double convolution block with batch normalization.

    Architecture:
        Conv2d(3x3) → BatchNorm2d → ReLU → Conv2d(3x3) → BatchNorm2d → ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels

    Example:
        >>> block = ConvBlock(3, 64)
        >>> x = torch.randn(1, 3, 128, 128)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 64, 128, 128])
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution block."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block with convolution and downsampling.

    Architecture:
        ConvBlock → MaxPool2d(2x2)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels

    Returns:
        Tuple of (downsampled_features, skip_connection)

    Example:
        >>> encoder = EncoderBlock(3, 64)
        >>> x = torch.randn(1, 3, 128, 128)
        >>> down, skip = encoder(x)
        >>> down.shape, skip.shape
        (torch.Size([1, 64, 64, 64]), torch.Size([1, 64, 128, 128]))
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder block.

        Returns:
            Tuple of (pooled_output, skip_connection)
        """
        skip = self.conv_block(x)
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connection.

    Architecture:
        ConvTranspose2d(2x2) → Concatenate with skip → ConvBlock

    Args:
        in_channels: Number of input channels (from previous decoder)
        skip_channels: Number of channels in skip connection
        out_channels: Number of output channels

    Example:
        >>> decoder = DecoderBlock(512, 256, 256)
        >>> x = torch.randn(1, 512, 16, 16)
        >>> skip = torch.randn(1, 256, 32, 32)
        >>> out = decoder(x, skip)
        >>> out.shape
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels, 
            in_channels // 2, 
            kernel_size=2, 
            stride=2
        )
        self.conv_block = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.

        Args:
            x: Input tensor from previous layer
            skip: Skip connection from encoder

        Returns:
            Upsampled and concatenated features
        """
        x = self.upconv(x)

        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                      diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for binary segmentation of brain tumors.

    Architecture:
        Encoder Path:
            Input (3, 128, 128)
            ↓ EncoderBlock(3 → 64)    → skip1 (64, 128, 128)
            ↓ EncoderBlock(64 → 128)  → skip2 (128, 64, 64)
            ↓ EncoderBlock(128 → 256) → skip3 (256, 32, 32)
            ↓ EncoderBlock(256 → 512) → skip4 (512, 16, 16)

        Bottleneck:
            ConvBlock(512 → 1024) at (1024, 8, 8)

        Decoder Path:
            ↑ DecoderBlock(1024 → 512) + skip4
            ↑ DecoderBlock(512 → 256)  + skip3
            ↑ DecoderBlock(256 → 128)  + skip2
            ↑ DecoderBlock(128 → 64)   + skip1

        Output:
            Conv2d(64 → 1) → Sigmoid → (1, 128, 128)

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary mask)
        features: Base number of features (default: 64)

    Example:
        >>> model = UNet(in_channels=3, out_channels=1)
        >>> x = torch.randn(4, 3, 128, 128)  # Batch of 4 images
        >>> output = model(x)
        >>> output.shape
        torch.Size([4, 1, 128, 128])
        >>> assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: int = 64):
        super(UNet, self).__init__()

        self.encoder1 = EncoderBlock(in_channels, features)
        self.encoder2 = EncoderBlock(features, features * 2)
        self.encoder3 = EncoderBlock(features * 2, features * 4)
        self.encoder4 = EncoderBlock(features * 4, features * 8)

        self.bottleneck = ConvBlock(features * 8, features * 16)

        self.decoder4 = DecoderBlock(features * 16, features * 8, features * 8)
        self.decoder3 = DecoderBlock(features * 8, features * 4, features * 4)
        self.decoder2 = DecoderBlock(features * 4, features * 2, features * 2)
        self.decoder1 = DecoderBlock(features * 2, features, features)

        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Segmentation mask of shape (batch_size, 1, H, W) with values in [0, 1]
        """
        enc1, skip1 = self.encoder1(x)
        enc2, skip2 = self.encoder2(enc1)
        enc3, skip3 = self.encoder3(enc2)
        enc4, skip4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck, skip4)
        dec3 = self.decoder3(dec4, skip3)
        dec2 = self.decoder2(dec3, skip2)
        dec1 = self.decoder1(dec2, skip1)

        output = self.output_conv(dec1)
        output = self.sigmoid(output)

        return output

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 128, 128)

    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Total parameters: {model.count_parameters():,}")
    print("✓ U-Net model loaded successfully!")