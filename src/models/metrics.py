"""
Evaluation metrics for segmentation models.

This module provides common segmentation metrics:
- Dice Coefficient (F1 Score)
- Intersection over Union (IoU / Jaccard Index)
- Pixel Accuracy
- Sensitivity (Recall)
- Specificity

All metrics support both binary and multi-class segmentation.

"""

import torch
import torch.nn.functional as F
from typing import Optional


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                     smooth: float = 1e-7) -> torch.Tensor:
    """
    Calculate Dice Coefficient (F1 Score) for segmentation.

    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)

    The Dice coefficient measures overlap between prediction and ground truth.
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
        target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}
        smooth: Smoothing constant to avoid division by zero (default: 1e-7)

    Returns:
        Dice coefficient averaged over batch

    Example:
        >>> pred = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        >>> dice = dice_coefficient(pred, target)
        >>> print(f"Dice: {dice:.3f}")
        Dice: 0.889
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, 
              smooth: float = 1e-7) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU / Jaccard Index).

    IoU = |X ∩ Y| / |X ∪ Y|

    Args:
        pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
        target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}
        smooth: Smoothing constant to avoid division by zero (default: 1e-7)

    Returns:
        IoU score averaged over batch

    Example:
        >>> pred = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        >>> target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        >>> iou = iou_score(pred, target)
        >>> print(f"IoU: {iou:.3f}")
        IoU: 1.000
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, 
                   threshold: float = 0.5) -> torch.Tensor:
    """
    Calculate pixel-wise accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
        target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}
        threshold: Threshold to binarize predictions (default: 0.5)

    Returns:
        Pixel accuracy averaged over batch

    Example:
        >>> pred = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        >>> acc = pixel_accuracy(pred, target)
        >>> print(f"Accuracy: {acc:.3f}")
        Accuracy: 1.000
    """
    pred_binary = (pred > threshold).float()

    correct = (pred_binary == target).float()
    accuracy = correct.mean()

    return accuracy


def sensitivity(pred: torch.Tensor, target: torch.Tensor, 
                threshold: float = 0.5, smooth: float = 1e-7) -> torch.Tensor:
    """
    Calculate Sensitivity (Recall / True Positive Rate).

    Sensitivity = TP / (TP + FN)

    Measures the proportion of actual positives correctly identified.
    Important for medical imaging to avoid missing tumors.

    Args:
        pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
        target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}
        threshold: Threshold to binarize predictions (default: 0.5)
        smooth: Smoothing constant to avoid division by zero (default: 1e-7)

    Returns:
        Sensitivity averaged over batch

    Example:
        >>> pred = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        >>> sens = sensitivity(pred, target)
        >>> print(f"Sensitivity: {sens:.3f}")
        Sensitivity: 1.000
    """
    pred_binary = (pred > threshold).float()

    pred_binary = pred_binary.view(pred_binary.size(0), -1)
    target = target.view(target.size(0), -1)

    tp = (pred_binary * target).sum(dim=1)
    actual_positive = target.sum(dim=1)

    sens = (tp + smooth) / (actual_positive + smooth)

    return sens.mean()


def specificity(pred: torch.Tensor, target: torch.Tensor, 
                threshold: float = 0.5, smooth: float = 1e-7) -> torch.Tensor:
    """
    Calculate Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)

    Measures the proportion of actual negatives correctly identified.

    Args:
        pred: Predicted mask of shape (batch_size, 1, H, W) with values in [0, 1]
        target: Ground truth mask of shape (batch_size, 1, H, W) with values in {0, 1}
        threshold: Threshold to binarize predictions (default: 0.5)
        smooth: Smoothing constant to avoid division by zero (default: 1e-7)

    Returns:
        Specificity averaged over batch

    Example:
        >>> pred = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        >>> spec = specificity(pred, target)
        >>> print(f"Specificity: {spec:.3f}")
        Specificity: 1.000
    """
    pred_binary = (pred > threshold).float()

    pred_binary = pred_binary.view(pred_binary.size(0), -1)
    target = target.view(target.size(0), -1)

    tn = ((1 - pred_binary) * (1 - target)).sum(dim=1)
    actual_negative = (1 - target).sum(dim=1)

    spec = (tn + smooth) / (actual_negative + smooth)

    return spec.mean()


class SegmentationMetrics:
    """
    Container for all segmentation metrics.

    Computes multiple metrics at once for convenience.

    Example:
        >>> metrics = SegmentationMetrics()
        >>> pred = torch.rand(4, 1, 128, 128)
        >>> target = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> results = metrics(pred, target)
        >>> print(results)
        {'dice': 0.45, 'iou': 0.35, 'accuracy': 0.82, 'sensitivity': 0.56, 'specificity': 0.89}
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics container.

        Args:
            threshold: Threshold for binarizing predictions (default: 0.5)
        """
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Calculate all metrics.

        Args:
            pred: Predicted mask of shape (batch_size, 1, H, W)
            target: Ground truth mask of shape (batch_size, 1, H, W)

        Returns:
            Dictionary with all metric values
        """
        return {
            'dice': dice_coefficient(pred, target).item(),
            'iou': iou_score(pred, target).item(),
            'accuracy': pixel_accuracy(pred, target, self.threshold).item(),
            'sensitivity': sensitivity(pred, target, self.threshold).item(),
            'specificity': specificity(pred, target, self.threshold).item()
        }


if __name__ == "__main__":
    pred = torch.rand(4, 1, 128, 128)
    target = torch.randint(0, 2, (4, 1, 128, 128)).float()

    metrics = SegmentationMetrics()
    results = metrics(pred, target)

    print("Segmentation Metrics Test:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

    print("✓ Metrics module loaded successfully!")
