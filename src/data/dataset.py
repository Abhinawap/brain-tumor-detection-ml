"""
PyTorch Dataset for brain tumor segmentation.

Loads brain tumor MRI images and segmentation masks, with optional
built-in augmentation for training.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from pathlib import Path


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor segmentation.

    Loads MRI images and corresponding segmentation masks from disk,
    resizes them, optionally applies augmentations, and returns PyTorch tensors.

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
          E.g., image/1/img.jpg -> mask/1/img.jpg or mask/1/img_m.jpg
          Images without a matching mask are skipped.

    Args:
        data_dir: Root directory containing 'image/' and 'mask/' subdirectories
        image_size: Target size for images (default: 128)
        classes: List of class indices to include (default: [0,1,2,3] for all)
        augment: Whether to apply random augmentations (default: False)
        aug_prob: Probability of applying each augmentation (default: 0.5)

    Example:
        >>> dataset = BrainTumorDataset(
        ...     data_dir='data/raw/Brain-Tumor-Segmentation-Dataset',
        ...     image_size=128,
        ...     classes=[1, 2, 3],  # Exclude "No Tumor"
        ...     augment=True
        ... )
        >>> image, mask = dataset[0]
        >>> image.shape, mask.shape
        (torch.Size([3, 128, 128]), torch.Size([1, 128, 128]))
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 128,
        classes: List[int] = [0, 1, 2, 3],
        augment: bool = False,
        aug_prob: float = 0.5
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.classes = classes
        self.augment = augment
        self.aug_prob = aug_prob

        # Load file paths
        self.image_paths, self.mask_paths = self._load_file_paths()

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}. Check directory structure.")

    def _find_mask_path(self, image_path: Path, mask_dir: Path) -> Optional[Path]:
        """
        Find corresponding mask file for an image.
        Handles masks with same name or with '_m' suffix.
        """
        for name in (image_path.name, image_path.stem + "_m" + image_path.suffix):
            mask_path = mask_dir / name
            if mask_path.exists():
                return mask_path
        return None

    def _load_file_paths(self) -> Tuple[List[Path], List[Path]]:
        """Load all image and mask file paths."""
        image_paths = []
        mask_paths = []

        for class_idx in self.classes:
            image_dir = self.data_dir / 'image' / str(class_idx)
            mask_dir = self.data_dir / 'mask' / str(class_idx)

            if not image_dir.exists():
                print(f"Warning: {image_dir} does not exist, skipping class {class_idx}")
                continue

            for img_path in sorted(image_dir.glob('*.jpg')):
                mask_path = self._find_mask_path(img_path, mask_dir)

                if mask_path is not None:
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)

        return image_paths, mask_paths

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.image_paths)

    def _apply_augmentations(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to image and mask."""

        if np.random.random() < self.aug_prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if np.random.random() < self.aug_prob:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        if np.random.random() < self.aug_prob:
            k = np.random.choice([1, 2, 3])
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        # Brightness adjustment
        if np.random.random() < self.aug_prob:
            alpha = np.random.uniform(0.8, 1.2)
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)

        return image, mask

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
        image = cv2.imread(str(self.image_paths[idx]))
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        image = image.astype(np.float32) / 255.0

        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
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
