"""
Loss functions for segmentation models.

- Dice Loss
- Combined BCE + Dice Loss (used for training)
"""

import torch
import torch.nn as nn

from src.models.metrics import dice_coefficient


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice Loss = 1 - Dice Coefficient (soft, on probabilities)

    Directly optimizes the Dice coefficient, so it handles class imbalance
    better than BCE alone.

    Args:
        smooth: Smoothing constant to avoid division by zero (default: 1e-7)

    Example:
        >>> criterion = DiceLoss()
        >>> pred = torch.rand(4, 1, 128, 128)
        >>> target = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> loss = criterion(pred, target)
    """

    def __init__(self, smooth: float = 1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - dice_coefficient(pred, target, self.smooth)


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.

    Loss = α * BCE + (1 - α) * Dice

    Combines pixel-level (BCE) and region-level (Dice) gradients.

    Args:
        alpha: Weight for BCE loss (default: 0.5)
               alpha = 0.5 means equal weighting
               alpha = 0.7 emphasizes BCE more
               alpha = 0.3 emphasizes Dice more
        smooth: Smoothing constant for Dice loss (default: 1e-7)

    Example:
        >>> criterion = BCEDiceLoss(alpha=0.5)
        >>> pred = torch.rand(4, 1, 128, 128)
        >>> target = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> loss = criterion(pred, target)
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1e-7):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * dice
