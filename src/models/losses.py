"""
Loss functions for segmentation models.

This module provides specialized loss functions for binary segmentation:
- Binary Cross-Entropy Loss (BCE)
- Dice Loss
- Combined BCE + Dice Loss (recommended for segmentation)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice Loss = 1 - Dice Coefficient

    This loss directly optimizes the Dice coefficient, making it ideal
    for segmentation tasks where class imbalance is present (e.g., small tumors).

    Args:
        smooth: Smoothing constant to avoid division by zero (default: 1e-7)

    Example:
        >>> criterion = DiceLoss()
        >>> pred = torch.rand(4, 1, 128, 128)
        >>> target = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> loss = criterion(pred, target)
        >>> print(f"Dice Loss: {loss.item():.4f}")
        Dice Loss: 0.4523
    """

    def __init__(self, smooth: float = 1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss.

        Args:
            pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
            target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}

        Returns:
            Dice loss (scalar)
        """
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1.0 - dice.mean()

        return dice_loss


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.

    Loss = α * BCE + (1 - α) * Dice

    This combined loss leverages:
    - BCE: Pixel-wise classification accuracy
    - Dice: Region overlap optimization

    The combination provides both pixel-level and region-level guidance,
    leading to better segmentation, especially for small objects.

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
        >>> print(f"Combined Loss: {loss.item():.4f}")
        Combined Loss: 0.6234
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1e-7):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined BCE + Dice loss.

        Args:
            pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
            target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}

        Returns:
            Combined loss (scalar)
        """
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)

        combined_loss = self.alpha * bce + (1 - self.alpha) * dice

        return combined_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    Focal loss down-weights easy examples and focuses on hard negatives.
    Useful when there's severe class imbalance (e.g., tiny tumors).

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               gamma = 0: reduces to BCE
               gamma > 0: focuses more on hard examples

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> pred = torch.rand(4, 1, 128, 128)
        >>> target = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> loss = criterion(pred, target)
        >>> print(f"Focal Loss: {loss.item():.4f}")
        Focal Loss: 0.1234
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss.

        Args:
            pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
            target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}

        Returns:
            Focal loss (scalar)
        """
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()


def get_loss_function(loss_name: str = 'bce_dice', **kwargs):
    """
    Factory function to get loss function by name.

    Args:
        loss_name: Name of loss function
                   Options: 'bce', 'dice', 'bce_dice', 'focal'
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance

    Example:
        >>> criterion = get_loss_function('bce_dice', alpha=0.5)
        >>> isinstance(criterion, BCEDiceLoss)
        True
    """
    loss_functions = {
        'bce': nn.BCELoss,
        'dice': DiceLoss,
        'bce_dice': BCEDiceLoss,
        'focal': FocalLoss
    }

    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: {list(loss_functions.keys())}")

    return loss_functions[loss_name](**kwargs)


if __name__ == "__main__":
    pred = torch.rand(4, 1, 128, 128)
    target = torch.randint(0, 2, (4, 1, 128, 128)).float()

    print("Loss Functions Test:")

    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"  Dice Loss: {loss.item():.4f}")

    bce_dice_loss = BCEDiceLoss(alpha=0.5)
    loss = bce_dice_loss(pred, target)
    print(f"  BCE+Dice Loss: {loss.item():.4f}")

    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(pred, target)
    print(f"  Focal Loss: {loss.item():.4f}")

    criterion = get_loss_function('bce_dice', alpha=0.7)
    print(f"  Factory: {type(criterion).__name__}")

    print("✓ Losses module loaded successfully!")
