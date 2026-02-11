"""
Data preprocessing module for brain tumor MRI scans.

This module provides preprocessing functions for MRI images including:
- Noise reduction (Wiener filtering via fastNlMeansDenoising)
- Contrast enhancement (CLAHE)
- Brain region extraction via contour detection
- Image normalization
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, List


def apply_wiener_filter(img: np.ndarray) -> np.ndarray:
    """
    Apply Wiener-like filtering using OpenCV's fastNlMeansDenoising.

    This function reduces noise while preserving edges, which is crucial
    for accurate brain region detection and tumor segmentation.

    Args:
        img: Input BGR image of shape (H, W, 3)

    Returns:
        Denoised BGR image of shape (H, W, 3)

    Example:
        >>> img = cv2.imread('mri_scan.jpg')
        >>> denoised = apply_wiener_filter(img)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.merge([denoised] * 3)


def apply_clahe(img: np.ndarray, clip_limit: float = 1.75, 
                tile_grid_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE enhances local contrast in the image, making tumor boundaries
    more distinguishable. The algorithm works in LAB color space to avoid
    affecting color information.

    Args:
        img: Input BGR image of shape (H, W, 3)
        clip_limit: Threshold for contrast limiting (default: 1.75)
        tile_grid_size: Size of grid for histogram equalization (default: 16x16)

    Returns:
        Contrast-enhanced BGR image of shape (H, W, 3)

    Example:
        >>> img = cv2.imread('mri_scan.jpg')
        >>> enhanced = apply_clahe(img)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    merged = cv2.merge([cl, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.

    This standardization is essential for neural network training,
    ensuring consistent input scales across all images.

    Args:
        img: Input image of shape (H, W, C) with values in [0, 255]

    Returns:
        Normalized image with values in [0, 1] and dtype float32

    Example:
        >>> img = cv2.imread('mri_scan.jpg')
        >>> normalized = normalize_image(img)
        >>> assert normalized.max() <= 1.0
        >>> assert normalized.dtype == np.float32
    """
    return img.astype('float32') / 255.0


def crop_brain_region(img: np.ndarray, mask: Optional[np.ndarray] = None, 
                     margin: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Crop the brain region from MRI scan using contour detection.

    This function removes black borders and focuses on the brain region,
    reducing noise and computational requirements. It uses thresholding
    and morphological operations to find the largest contour (brain).

    Args:
        img: Input BGR image of shape (H, W, 3)
        mask: Optional corresponding mask of shape (H, W) or (H, W, 1)
        margin: Additional pixels to include around detected region (default: 0)

    Returns:
        Tuple of (cropped_image, cropped_mask) where:
        - cropped_image: Brain region of shape (H', W', 3)
        - cropped_mask: Cropped mask of shape (H', W', 1) or None if mask not provided

    Example:
        >>> img = cv2.imread('mri_scan.jpg')
        >>> mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
        >>> cropped_img, cropped_mask = crop_brain_region(img, mask)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not cnts:
        return (img, mask) if mask is not None else (img, None)

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, img.shape[1] - x)
    h = min(h + 2 * margin, img.shape[0] - y)

    img_cropped = img[y:y+h, x:x+w]

    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        mask_cropped = mask[y:y+h, x:x+w]
        return img_cropped, mask_cropped

    return img_cropped, None


def load_data(base_path: str, target_size: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess brain tumor dataset.

    Dataset structure expected:
        base_path/
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

    Args:
        base_path: Root directory containing image/ and mask/ folders
        target_size: Desired output size (H, W) for all images (default: 128x128)

    Returns:
        Tuple of (images, masks) where:
        - images: numpy array of shape (N, H, W, 3) with values in [0, 1]
        - masks: numpy array of shape (N, H, W, 1) with values in [0, 1]

    Example:
        >>> X, Y = load_data('../data/raw/Brain Tumor Segmentation Dataset')
        >>> print(f"Loaded {len(X)} images")
        >>> print(f"Image shape: {X[0].shape}, Mask shape: {Y[0].shape}")
    """
    images = []
    masks = []

    for label in range(4):
        image_folder = os.path.join(base_path, 'image', str(label))
        mask_folder = os.path.join(base_path, 'mask', str(label))

        if not os.path.exists(image_folder):
            print(f"Warning: {image_folder} does not exist, skipping...")
            continue

        for filename in os.listdir(image_folder):
            img_path = os.path.join(image_folder, filename)

            file_root, _ = os.path.splitext(filename)
            mask_path = os.path.join(mask_folder, f"{file_root}_m.jpg")

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            if np.max(mask) == 0:
                continue

            img, mask = crop_brain_region(img, mask)

            if img is None or img.size == 0 or mask is None or mask.size == 0:
                continue

            img = cv2.resize(img, target_size)
            mask = cv2.resize(mask, target_size)

            img = apply_wiener_filter(img)
            img = apply_clahe(img)
            img = normalize_image(img)

            mask = (mask > 0).astype(np.float32)[..., np.newaxis]

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)


if __name__ == "__main__":
    print("Preprocessing module loaded successfully!")
    print("Available functions:")
    print("  - apply_wiener_filter()")
    print("  - apply_clahe()")
    print("  - normalize_image()")
    print("  - crop_brain_region()")
    print("  - load_data()")