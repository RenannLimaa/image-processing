"""
Pre-processing module — spatial domain transformations.

All core operations are implemented manually (no cv2.GaussianBlur,
cv2.equalizeHist etc.).  OpenCV is used only for imread / imwrite / resize.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image from disk in BGR format (OpenCV default)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """Save an ndarray image to disk."""
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# Spatial transformations (manual)
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the standard luminosity weights:
        Y = 0.299 R + 0.587 G + 0.114 B
    Returns a 2-D uint8 array.
    """
    if img.ndim == 2:
        return img  # already grayscale
    b = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    r = img[:, :, 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def resize_image(img: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize image to (width, height).  cv2.resize is allowed for scaling."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Build a normalised 2-D Gaussian kernel manually."""
    k = kernel_size // 2
    y, x = np.mgrid[-k : k + 1, -k : k + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2-D convolution with zero-padding.
    Operates on a single-channel float64 image.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (padded[i : i + kh, j : j + kw] * kernel).sum()
    return out


def gaussian_blur(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Apply Gaussian blur using manual 2-D convolution.
    Accepts grayscale (2-D) or BGR (3-D) images.
    """
    kernel = _gaussian_kernel(kernel_size, sigma)
    if img.ndim == 2:
        blurred = _convolve2d(img, kernel)
        return np.clip(blurred, 0, 255).astype(np.uint8)
    # Multi-channel: blur each channel independently
    channels = [_convolve2d(img[:, :, c], kernel) for c in range(img.shape[2])]
    blurred = np.stack(channels, axis=-1)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def histogram_equalize(img: np.ndarray) -> np.ndarray:
    """
    Manual histogram equalization for a grayscale uint8 image.

    Steps:
      1. Compute pixel frequency histogram.
      2. Compute CDF.
      3. Normalise CDF to [0, 255].
      4. Apply mapping as a look-up table.
    """
    if img.ndim != 2:
        raise ValueError("histogram_equalize expects a single-channel grayscale image.")
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    # Mask zero-frequency bins to avoid division issues
    cdf_min = cdf[cdf > 0].min()
    total_pixels = img.size
    lut = np.round(
        (cdf - cdf_min) / (total_pixels - cdf_min) * 255
    ).astype(np.uint8)
    return lut[img]


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalise pixel values to [0, 1] float64."""
    return img.astype(np.float64) / 255.0


def preprocess(path: str, size: tuple[int, int] = (256, 256)) -> dict:
    """
    Full pre-processing pipeline for a single image.

    Returns a dict with keys:
        original, gray, resized, blurred, equalized
    """
    original = load_image(path)
    gray = to_grayscale(original)
    resized = resize_image(gray, size)
    blurred = gaussian_blur(resized, kernel_size=5, sigma=1.4)
    equalized = histogram_equalize(blurred)
    return {
        "original": original,
        "gray": gray,
        "resized": resized,
        "blurred": blurred,
        "equalized": equalized,
    }
