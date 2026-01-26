"""
Unit tests for PyTorch Dataset classes.

Tests cover:
- Dataset initialization and loading
- Data transformations
- Augmentations
- Tensor shapes and types
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.data.dataset import BrainTumorDataset, BrainTumorDatasetWithAugmentation


class TestBrainTumorDataset:
    """Tests for BrainTumorDataset."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory structure."""
        # Create directory structure
        for class_idx in [0, 1, 2, 3]:
            (tmp_path / 'image' / str(class_idx)).mkdir(parents=True)
            (tmp_path / 'mask' / str(class_idx)).mkdir(parents=True)

        # Create dummy images
        import cv2
        for class_idx in [1, 2, 3]:  # Create a few samples
            for i in range(3):
                # Image
                img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                img_path = tmp_path / 'image' / str(class_idx) / f'test_{i}.jpg'
                cv2.imwrite(str(img_path), img)

                # Mask
                mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8) * 255
                mask_path = tmp_path / 'mask' / str(class_idx) / f'test_{i}.jpg'
                cv2.imwrite(str(mask_path), mask)

        return tmp_path

    def test_dataset_initialization(self, mock_dataset_dir):
        """Test dataset can be initialized."""
        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            image_size=128,
            classes=[1, 2, 3]
        )

        assert len(dataset) == 9  # 3 classes × 3 samples

    def test_dataset_length(self, mock_dataset_dir):
        """Test dataset __len__ method."""
        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            image_size=128,
            classes=[1, 2]
        )

        assert len(dataset) == 6  # 2 classes × 3 samples

    def test_dataset_getitem(self, mock_dataset_dir):
        """Test dataset __getitem__ method."""
        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            image_size=128,
            classes=[1, 2, 3]
        )

        image, mask = dataset[0]

        # Check types
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

        # Check shapes
        assert image.shape == (3, 128, 128)
        assert mask.shape == (1, 128, 128)

        # Check dtypes
        assert image.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_dataset_image_normalization(self, mock_dataset_dir):
        """Test that images are normalized to [0, 1]."""
        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            image_size=128,
            classes=[1, 2, 3]
        )

        image, mask = dataset[0]

        # Image should be in [0, 1]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_dataset_mask_binarization(self, mock_dataset_dir):
        """Test that masks are binarized to {0, 1}."""
        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            image_size=128,
            classes=[1, 2, 3]
        )

        image, mask = dataset[0]

        # Mask should only contain 0 and 1
        unique_values = mask.unique().tolist()
        assert all(val in [0.0, 1.0] for val in unique_values)

    def test_dataset_image_resize(self, mock_dataset_dir):
        """Test that images are resized correctly."""
        for size in [64, 128, 256]:
            dataset = BrainTumorDataset(
                data_dir=str(mock_dataset_dir),
                image_size=size,
                classes=[1]
            )

            image, mask = dataset[0]

            assert image.shape == (3, size, size)
            assert mask.shape == (1, size, size)

    def test_dataset_class_distribution(self, mock_dataset_dir):
        """Test get_class_distribution method."""
        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            image_size=128,
            classes=[1, 2, 3]
        )

        dist = dataset.get_class_distribution()

        # Should have 3 samples per class
        assert dist[1] == 3
        assert dist[2] == 3
        assert dist[3] == 3

    def test_dataset_empty_raises_error(self, tmp_path):
        """Test that empty dataset raises error."""
        # Create empty directory structure
        (tmp_path / 'image' / '1').mkdir(parents=True)
        (tmp_path / 'mask' / '1').mkdir(parents=True)

        with pytest.raises(ValueError, match="No images found"):
            BrainTumorDataset(
                data_dir=str(tmp_path),
                classes=[1]
            )


class TestBrainTumorDatasetWithAugmentation:
    """Tests for BrainTumorDatasetWithAugmentation."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory structure."""
        import cv2

        # Create directory structure
        for class_idx in [1, 2, 3]:
            (tmp_path / 'image' / str(class_idx)).mkdir(parents=True)
            (tmp_path / 'mask' / str(class_idx)).mkdir(parents=True)

            # Create a few samples
            for i in range(3):
                img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                img_path = tmp_path / 'image' / str(class_idx) / f'test_{i}.jpg'
                cv2.imwrite(str(img_path), img)

                mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8) * 255
                mask_path = tmp_path / 'mask' / str(class_idx) / f'test_{i}.jpg'
                cv2.imwrite(str(mask_path), mask)

        return tmp_path

    def test_augmented_dataset_initialization(self, mock_dataset_dir):
        """Test augmented dataset initialization."""
        dataset = BrainTumorDatasetWithAugmentation(
            data_dir=str(mock_dataset_dir),
            augment=True,
            aug_prob=0.5
        )

        assert len(dataset) == 9
        assert dataset.augment is True

    def test_augmentation_disabled(self, mock_dataset_dir):
        """Test that augmentation can be disabled."""
        dataset = BrainTumorDatasetWithAugmentation(
            data_dir=str(mock_dataset_dir),
            augment=False
        )

        # Get same sample twice
        image1, mask1 = dataset[0]
        image2, mask2 = dataset[0]

        # Should be identical when augmentation is off
        assert torch.allclose(image1, image2)
        assert torch.allclose(mask1, mask2)

    def test_augmentation_enabled(self, mock_dataset_dir):
        """Test that augmentation produces different outputs."""
        dataset = BrainTumorDatasetWithAugmentation(
            data_dir=str(mock_dataset_dir),
            augment=True,
            aug_prob=1.0  # Force augmentation
        )

        # Get same sample multiple times
        samples = [dataset[0] for _ in range(5)]

        # At least some should be different due to augmentation
        # (with aug_prob=1.0, all augmentations apply, so outputs will differ)
        images = [s[0] for s in samples]

        # Check that not all images are identical
        all_same = all(torch.allclose(images[0], img) for img in images[1:])
        assert not all_same  # Should have variation due to augmentation

    def test_augmentation_preserves_shape(self, mock_dataset_dir):
        """Test that augmentation preserves tensor shapes."""
        dataset = BrainTumorDatasetWithAugmentation(
            data_dir=str(mock_dataset_dir),
            augment=True,
            aug_prob=1.0
        )

        image, mask = dataset[0]

        assert image.shape == (3, 128, 128)
        assert mask.shape == (1, 128, 128)


class TestDatasetIntegration:
    """Integration tests for Dataset with DataLoader."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create mock dataset."""
        import cv2

        for class_idx in [1, 2]:
            (tmp_path / 'image' / str(class_idx)).mkdir(parents=True)
            (tmp_path / 'mask' / str(class_idx)).mkdir(parents=True)

            for i in range(5):
                img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                img_path = tmp_path / 'image' / str(class_idx) / f'test_{i}.jpg'
                cv2.imwrite(str(img_path), img)

                mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8) * 255
                mask_path = tmp_path / 'mask' / str(class_idx) / f'test_{i}.jpg'
                cv2.imwrite(str(mask_path), mask)

        return tmp_path

    def test_dataloader_integration(self, mock_dataset_dir):
        """Test dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            image_size=128,
            classes=[1, 2]
        )

        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Get one batch
        images, masks = next(iter(loader))

        # Check batch dimensions
        assert images.shape == (4, 3, 128, 128)
        assert masks.shape == (4, 1, 128, 128)

    def test_dataloader_iteration(self, mock_dataset_dir):
        """Test iterating through entire dataset."""
        from torch.utils.data import DataLoader

        dataset = BrainTumorDataset(
            data_dir=str(mock_dataset_dir),
            classes=[1, 2]
        )

        loader = DataLoader(dataset, batch_size=2)

        total_samples = 0
        for images, masks in loader:
            total_samples += images.shape[0]

        assert total_samples == len(dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])