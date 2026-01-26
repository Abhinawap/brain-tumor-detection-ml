"""
Unit tests for U-Net model, metrics, and losses.

Tests cover:
- U-Net architecture forward pass
- Model parameter count
- Segmentation metrics (Dice, IoU, accuracy, etc.)
- Loss functions (Dice, BCE+Dice, Focal)
"""

import pytest
import torch
import torch.nn as nn
from src.models.unet import UNet, ConvBlock, EncoderBlock, DecoderBlock
from src.models.metrics import (
    dice_coefficient, 
    iou_score, 
    pixel_accuracy,
    sensitivity,
    specificity,
    SegmentationMetrics
)
from src.models.losses import DiceLoss, BCEDiceLoss, FocalLoss, get_loss_function


class TestUNetArchitecture:
    """Tests for U-Net model architecture."""

    def test_unet_output_shape(self):
        """Test that U-Net produces correct output shape."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 128, 128)

        output = model(x)

        assert output.shape == (2, 1, 128, 128)

    def test_unet_output_range(self):
        """Test that U-Net output is in [0, 1] range (sigmoid)."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 128, 128)

        output = model(x)

        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_unet_different_input_sizes(self):
        """Test U-Net with different input sizes."""
        model = UNet(in_channels=3, out_channels=1)

        # Test multiple sizes
        for size in [64, 128, 256]:
            x = torch.randn(1, 3, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)

    def test_unet_batch_processing(self):
        """Test U-Net with different batch sizes."""
        model = UNet(in_channels=3, out_channels=1)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 128, 128)
            output = model(x)
            assert output.shape[0] == batch_size

    def test_unet_parameter_count(self):
        """Test that U-Net has reasonable number of parameters."""
        model = UNet(in_channels=3, out_channels=1, features=64)
        param_count = model.count_parameters()

        # U-Net should have millions of parameters but not too many
        assert 1_000_000 < param_count < 100_000_000

    def test_unet_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 128, 128, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None


class TestUNetComponents:
    """Tests for U-Net building blocks."""

    def test_conv_block(self):
        """Test ConvBlock forward pass."""
        block = ConvBlock(3, 64)
        x = torch.randn(2, 3, 128, 128)

        output = block(x)

        assert output.shape == (2, 64, 128, 128)

    def test_encoder_block(self):
        """Test EncoderBlock produces correct outputs."""
        encoder = EncoderBlock(3, 64)
        x = torch.randn(2, 3, 128, 128)

        down, skip = encoder(x)

        # Downsampled output is half size
        assert down.shape == (2, 64, 64, 64)
        # Skip connection preserves input size
        assert skip.shape == (2, 64, 128, 128)

    def test_decoder_block(self):
        """Test DecoderBlock with skip connection."""
        decoder = DecoderBlock(128, 64, 64)
        x = torch.randn(2, 128, 32, 32)
        skip = torch.randn(2, 64, 64, 64)

        output = decoder(x, skip)

        # Output should match skip connection size
        assert output.shape == (2, 64, 64, 64)


class TestSegmentationMetrics:
    """Tests for segmentation metrics."""

    def test_dice_coefficient_perfect(self):
        """Test Dice coefficient with perfect prediction."""
        pred = torch.ones(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64)

        dice = dice_coefficient(pred, target)

        assert torch.isclose(dice, torch.tensor(1.0), atol=1e-5)

    def test_dice_coefficient_zero(self):
        """Test Dice coefficient with no overlap."""
        pred = torch.ones(2, 1, 64, 64)
        target = torch.zeros(2, 1, 64, 64)

        dice = dice_coefficient(pred, target)

        # Should be close to 0 (with smoothing)
        assert dice < 0.01

    def test_iou_score_perfect(self):
        """Test IoU score with perfect prediction."""
        pred = torch.ones(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64)

        iou = iou_score(pred, target)

        assert torch.isclose(iou, torch.tensor(1.0), atol=1e-5)

    def test_pixel_accuracy(self):
        """Test pixel accuracy calculation."""
        pred = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

        acc = pixel_accuracy(pred, target, threshold=0.5)

        # All 4 pixels should be correct
        assert torch.isclose(acc, torch.tensor(1.0))

    def test_sensitivity(self):
        """Test sensitivity (recall) calculation."""
        # Perfect sensitivity case
        pred = torch.ones(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64)

        sens = sensitivity(pred, target, threshold=0.5)

        assert torch.isclose(sens, torch.tensor(1.0), atol=1e-5)

    def test_specificity(self):
        """Test specificity calculation."""
        # Perfect specificity case
        pred = torch.zeros(2, 1, 64, 64)
        target = torch.zeros(2, 1, 64, 64)

        spec = specificity(pred, target, threshold=0.5)

        assert torch.isclose(spec, torch.tensor(1.0), atol=1e-5)

    def test_segmentation_metrics_container(self):
        """Test SegmentationMetrics container."""
        metrics = SegmentationMetrics(threshold=0.5)
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        results = metrics(pred, target)

        # Check all metrics are present
        assert 'dice' in results
        assert 'iou' in results
        assert 'accuracy' in results
        assert 'sensitivity' in results
        assert 'specificity' in results

        # Check all values are valid
        for value in results.values():
            assert 0.0 <= value <= 1.0


class TestLossFunctions:
    """Tests for loss functions."""

    def test_dice_loss_perfect(self):
        """Test Dice loss with perfect prediction."""
        criterion = DiceLoss()
        pred = torch.ones(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64)

        loss = criterion(pred, target)

        # Perfect prediction should have loss close to 0
        assert loss < 0.01

    def test_dice_loss_worst(self):
        """Test Dice loss with worst prediction."""
        criterion = DiceLoss()
        pred = torch.ones(2, 1, 64, 64)
        target = torch.zeros(2, 1, 64, 64)

        loss = criterion(pred, target)

        # Worst prediction should have loss close to 1
        assert loss > 0.99

    def test_bce_dice_loss(self):
        """Test combined BCE + Dice loss."""
        criterion = BCEDiceLoss(alpha=0.5)
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = criterion(pred, target)

        # Loss should be scalar and positive
        assert loss.ndim == 0
        assert loss > 0

    def test_bce_dice_loss_weights(self):
        """Test BCE+Dice loss with different alpha values."""
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Different alpha values should produce different losses
        loss_0 = BCEDiceLoss(alpha=0.0)(pred, target)
        loss_5 = BCEDiceLoss(alpha=0.5)(pred, target)
        loss_1 = BCEDiceLoss(alpha=1.0)(pred, target)

        # All should be different
        assert not torch.isclose(loss_0, loss_5)
        assert not torch.isclose(loss_5, loss_1)

    def test_focal_loss(self):
        """Test Focal loss."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = criterion(pred, target)

        # Loss should be scalar and positive
        assert loss.ndim == 0
        assert loss > 0

    def test_loss_factory(self):
        """Test loss function factory."""
        # Test all supported loss functions
        for loss_name in ['bce', 'dice', 'bce_dice', 'focal']:
            criterion = get_loss_function(loss_name)
            assert criterion is not None

    def test_loss_factory_invalid(self):
        """Test loss factory with invalid name."""
        with pytest.raises(ValueError):
            get_loss_function('invalid_loss')


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_training_step(self):
        """Test a complete training step."""
        # Setup
        model = UNet(in_channels=3, out_channels=1)
        criterion = BCEDiceLoss(alpha=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Forward pass
        x = torch.rand(2, 3, 128, 128)
        target = torch.randint(0, 2, (2, 1, 128, 128)).float()

        pred = model(x)
        loss = criterion(pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check everything ran without errors
        assert loss.item() > 0

    def test_evaluation_step(self):
        """Test a complete evaluation step."""
        model = UNet(in_channels=3, out_channels=1)
        metrics = SegmentationMetrics()

        model.eval()
        with torch.no_grad():
            x = torch.rand(2, 3, 128, 128)
            target = torch.randint(0, 2, (2, 1, 128, 128)).float()

            pred = model(x)
            results = metrics(pred, target)

        # Check all metrics are computed
        assert len(results) == 5
        assert all(0 <= v <= 1 for v in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])