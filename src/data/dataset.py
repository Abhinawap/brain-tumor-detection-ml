"""
PyTorch Dataset for brain tumor segmentation.

This module provides Dataset classes for loading and preprocessing
brain tumor MRI images and segmentation masks.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Callable, List
from pathlib import Path


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor segmentation.

    Loads MRI images and corresponding segmentation masks from disk,
    applies preprocessing and augmentations, and returns PyTorch tensors.

    Directory structure expected:
        data_dir/
        ├── image/
        │   ├── 0/  (No Tumor)
        │   ├── 1/  (Glioma)
        │   ├── 2/  (Meningioma)
        │   └── 3/  (Pituitary)
        └── mask/
            ├── 0/
            ├── 1/
            ├── 2/
            └── 3/

    Note: Masks should have same filename as images, optionally with '_m' suffix.
          E.g., image/0/img.jpg -> mask/0/img.jpg or mask/0/img_m.jpg

    Args:
        data_dir: Root directory containing 'image/' and 'mask/' subdirectories
        transform: Optional transform to apply to both image and mask
        image_size: Target size for images (default: 128)
        classes: List of class indices to include (default: [0,1,2,3] for all)

    Example:
        >>> dataset = BrainTumorDataset(
        ...     data_dir='data/raw/Brain-Tumor-Segmentation-Dataset',
        ...     image_size=128,
        ...     classes=[1, 2, 3]  # Exclude "No Tumor"
        ... )
        >>> len(dataset)
        2500
        >>> image, mask = dataset[0]
        >>> image.shape, mask.shape
        (torch.Size([3, 128, 128]), torch.Size([1, 128, 128]))
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 128,
        classes: List[int] = [0, 1, 2, 3]
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        self.classes = classes

        # Load file paths
        self.image_paths, self.mask_paths = self._load_file_paths()

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}. Check directory structure.")

    def _find_mask_path(self, image_path: Path, mask_dir: Path) -> Optional[Path]:
        """
        Find corresponding mask file for an image.
        Handles masks with same name or with '_m' suffix.
        """
        # Try exact match first
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            return mask_path

        # Try with _m suffix
        mask_name = image_path.stem + "_m" + image_path.suffix
        mask_path = mask_dir / mask_name
        if mask_path.exists():
            return mask_path

        # Try .jpg extension if original was different
        mask_path = mask_dir / (image_path.stem + "_m.jpg")
        if mask_path.exists():
            return mask_path

        return None

    def _load_file_paths(self) -> Tuple[List[Path], List[Path]]:
        """Load all image and mask file paths."""
        image_paths = []
        mask_paths = []

        for class_idx in self.classes:
            # Image directory for this class
            image_dir = self.data_dir / 'image' / str(class_idx)
            mask_dir = self.data_dir / 'mask' / str(class_idx)

            if not image_dir.exists():
                print(f"Warning: {image_dir} does not exist, skipping class {class_idx}")
                continue

            # Get all image files
            for img_path in sorted(image_dir.glob('*.jpg')):
                # Find corresponding mask
                mask_path = self._find_mask_path(img_path, mask_dir)

                if mask_path is not None:
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)

        return image_paths, mask_paths

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return image and mask at given index.

        Args:
            idx: Index of sample to load

        Returns:
            Tuple of (image, mask) as PyTorch tensors
            - image: (3, H, W) float32 tensor, values in [0, 1]
            - mask: (1, H, W) float32 tensor, values in {0, 1}
        """
        # Load image and mask
        image = cv2.imread(str(self.image_paths[idx]))
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Binarize mask (any non-zero value becomes 1)
        mask = (mask > 0).astype(np.float32)

        # Convert to PyTorch tensors
        # Image: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Mask: (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

    def get_class_distribution(self) -> dict:
        """
        Get distribution of samples across classes.

        Returns:
            Dictionary mapping class index to count
        """
        distribution = {cls: 0 for cls in self.classes}

        for img_path in self.image_paths:
            class_idx = int(img_path.parent.name)
            distribution[class_idx] += 1

        return distribution


class BrainTumorDatasetWithAugmentation(BrainTumorDataset):
    """
    BrainTumorDataset with built-in augmentation support.

    This extends the base dataset to include common augmentations
    without requiring external transform libraries.

    Args:
        data_dir: Root directory
        image_size: Target image size
        classes: Classes to include
        augment: Whether to apply augmentations (default: True)
        aug_prob: Probability of applying each augmentation (default: 0.5)

    Example:
        >>> dataset = BrainTumorDatasetWithAugmentation(
        ...     data_dir='data/raw/Brain-Tumor-Segmentation-Dataset',
        ...     augment=True,
        ...     aug_prob=0.5
        ... )
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 128,
        classes: List[int] = [0, 1, 2, 3],
        augment: bool = True,
        aug_prob: float = 0.5
    ):
        super().__init__(data_dir, transform=None, image_size=image_size, classes=classes)
        self.augment = augment
        self.aug_prob = aug_prob

    def _apply_augmentations(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to image and mask."""

        # Horizontal flip
        if np.random.random() < self.aug_prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Vertical flip
        if np.random.random() < self.aug_prob:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Rotation (90, 180, 270 degrees)
        if np.random.random() < self.aug_prob:
            k = np.random.choice([1, 2, 3])  # 90, 180, 270 degrees
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        # Brightness adjustment
        if np.random.random() < self.aug_prob:
            alpha = np.random.uniform(0.8, 1.2)  # Brightness factor
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load image and mask with augmentations."""
        # Load image and mask
        image = cv2.imread(str(self.image_paths[idx]))
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # Apply augmentations if enabled
        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


if __name__ == "__main__":
    # Quick test
    data_dir = "../data/raw/Brain-Tumor-Segmentation-Dataset"

    if os.path.exists(data_dir):
        dataset = BrainTumorDataset(data_dir, image_size=128)
        print(f"Dataset loaded: {len(dataset)} samples")

        # Test loading one sample
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Mask unique values: {mask.unique().tolist()}")

        # Class distribution
        dist = dataset.get_class_distribution()
        print(f"Class distribution: {dist}")

        print("✓ Dataset module loaded successfully!")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Dataset module loaded (untested)")