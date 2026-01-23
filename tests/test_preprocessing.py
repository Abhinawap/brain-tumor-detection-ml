"""
Unit tests for data preprocessing module.

Tests cover:
- Image normalization
- CLAHE enhancement
- Brain region cropping
- Wiener filtering

Run with: pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import cv2
from src.data.preprocessing import (
    normalize_image,
    apply_clahe,
    apply_wiener_filter,
    crop_brain_region
)


class TestNormalization:
    """Tests for image normalization."""

    def test_normalize_image_range(self):
        """Test that normalized values are in [0, 1] range."""
        # Create image with varying values (0 to 255)
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        normalized = normalize_image(img)
        
        # Check values are in valid range
        assert 0.0 <= normalized.min() <= 1.0
        assert 0.0 <= normalized.max() <= 1.0
        assert normalized.max() > normalized.min()

    def test_normalize_image_dtype(self):
        """Test that output dtype is float32."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        normalized = normalize_image(img)

        assert normalized.dtype == np.float32

    def test_normalize_image_shape(self):
        """Test that shape is preserved."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        normalized = normalize_image(img)

        assert normalized.shape == img.shape


class TestCLAHE:
    """Tests for CLAHE enhancement."""

    def test_clahe_output_shape(self):
        """Test that CLAHE preserves image shape."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        enhanced = apply_clahe(img)

        assert enhanced.shape == img.shape

    def test_clahe_output_dtype(self):
        """Test that CLAHE preserves dtype."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        enhanced = apply_clahe(img)

        assert enhanced.dtype == img.dtype

    def test_clahe_custom_parameters(self):
        """Test CLAHE with custom parameters."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        enhanced = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))

        assert enhanced.shape == img.shape
        assert enhanced.dtype == img.dtype


class TestWienerFilter:
    """Tests for Wiener filtering."""

    def test_wiener_output_shape(self):
        """Test that Wiener filter preserves shape."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        filtered = apply_wiener_filter(img)

        assert filtered.shape == img.shape

    def test_wiener_output_channels(self):
        """Test that output has 3 channels."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        filtered = apply_wiener_filter(img)

        assert filtered.shape[2] == 3


class TestBrainCropping:
    """Tests for brain region cropping."""

    def test_crop_without_mask(self):
        """Test cropping without mask."""
        # Create synthetic brain-like image (white circle on black background)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(img, (128, 128), 80, (200, 200, 200), -1)

        cropped, mask_out = crop_brain_region(img, mask=None)

        assert cropped.shape[0] < img.shape[0]  # Should be smaller
        assert cropped.shape[1] < img.shape[1]
        assert mask_out is None

    def test_crop_with_mask(self):
        """Test cropping with mask."""
        # Create synthetic brain image and mask
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256, 1), dtype=np.uint8)
        cv2.circle(img, (128, 128), 80, (200, 200, 200), -1)
        cv2.circle(mask, (128, 128), 40, (255, 255, 255), -1)

        cropped_img, cropped_mask = crop_brain_region(img, mask)

        assert cropped_img.shape[0] < img.shape[0]
        assert cropped_mask is not None
        assert cropped_mask.shape[0] == cropped_img.shape[0]
        assert cropped_mask.shape[1] == cropped_img.shape[1]

    def test_crop_with_margin(self):
        """Test cropping with additional margin."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(img, (128, 128), 80, (200, 200, 200), -1)

        cropped_no_margin, _ = crop_brain_region(img, margin=0)
        cropped_with_margin, _ = crop_brain_region(img, margin=10)

        # With margin should be slightly larger
        assert cropped_with_margin.shape[0] >= cropped_no_margin.shape[0]
        assert cropped_with_margin.shape[1] >= cropped_no_margin.shape[1]

    def test_crop_empty_image(self):
        """Test cropping on completely black image."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        cropped, _ = crop_brain_region(img)

        # Should return original image if no contours found
        assert cropped.shape == img.shape


class TestIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing workflow."""
        # Create synthetic MRI image
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        cv2.circle(img, (128, 128), 80, (180, 180, 180), -1)

        # Apply full pipeline
        img_cropped, _ = crop_brain_region(img)
        img_filtered = apply_wiener_filter(img_cropped)
        img_enhanced = apply_clahe(img_filtered)
        img_normalized = normalize_image(img_enhanced)

        # Final checks
        assert img_normalized.dtype == np.float32
        assert 0.0 <= img_normalized.min() <= img_normalized.max() <= 1.0
        assert img_normalized.shape[2] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
