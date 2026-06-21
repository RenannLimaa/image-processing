"""
Edges and segmentation module.

Implements:
  - Manual Canny edge detector (Sobel gradients → NMS → hysteresis)
  - Manual Otsu thresholding

No cv2.Canny, cv2.Sobel, or cv2.threshold(OTSU) are used.
"""

import numpy as np
from src.preprocessing import _convolve2d


# ---------------------------------------------------------------------------
# Sobel gradients
# ---------------------------------------------------------------------------

SOBEL_GX = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)

SOBEL_GY = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float64)


def sobel_gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and direction using 3×3 Sobel kernels.

    Returns
    -------
    magnitude  : float64 array, same shape as img
    direction  : float64 array of angles in degrees [0, 180)
    """
    gx = _convolve2d(img.astype(np.float64), SOBEL_GX)
    gy = _convolve2d(img.astype(np.float64), SOBEL_GY)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.degrees(np.arctan2(gy, gx)) % 180  # map to [0, 180)
    return magnitude, direction


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Thin edges to 1-pixel width by suppressing non-maximum gradient pixels.

    Gradient direction is quantised to 0°, 45°, 90°, or 135° and each pixel
    is compared against its two neighbours along the gradient.
    """
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = direction[i, j]
            m = magnitude[i, j]

            # Quantise angle and pick neighbours
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= angle < 67.5:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            elif 67.5 <= angle < 112.5:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:  # 112.5 <= angle < 157.5
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]

            if m >= n1 and m >= n2:
                suppressed[i, j] = m

    return suppressed


# ---------------------------------------------------------------------------
# Hysteresis thresholding
# ---------------------------------------------------------------------------

def hysteresis(img: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15) -> np.ndarray:
    """
    Apply double thresholding and edge-tracking by hysteresis.

    Thresholds are set as fractions of the maximum gradient magnitude:
        high = high_ratio * max
        low  = low_ratio  * max

    Returns a binary edge map (uint8: 0 or 255).
    """
    max_val = img.max()
    if max_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    high = high_ratio * max_val
    low = low_ratio * max_val

    strong = (img >= high).astype(np.uint8)
    weak = ((img >= low) & (img < high)).astype(np.uint8)

    # Track: a weak pixel becomes strong if it touches a strong pixel (8-connectivity)
    result = strong.copy()
    changed = True
    while changed:
        changed = False
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if weak[i, j] == 1 and result[i, j] == 0:
                    neighbours = result[i - 1 : i + 2, j - 1 : j + 2]
                    if neighbours.max() > 0:
                        result[i, j] = 1
                        changed = True

    return (result * 255).astype(np.uint8)


def canny(
    img: np.ndarray,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
) -> np.ndarray:
    """
    Full manual Canny edge detection pipeline:
      1. Sobel gradients
      2. Non-maximum suppression
      3. Hysteresis thresholding

    Returns a binary edge map (uint8: 0 or 255).
    """
    magnitude, direction = sobel_gradient(img.astype(np.float64))
    suppressed = non_max_suppression(magnitude, direction)
    edges = hysteresis(suppressed, low_ratio=low_ratio, high_ratio=high_ratio)
    return edges


def edge_density(edge_map: np.ndarray) -> float:
    """
    Feature: ratio of edge pixels to total pixels.
    Higher → more edges (cracks, irregular surface).
    """
    return float((edge_map > 0).sum() / edge_map.size)


# ---------------------------------------------------------------------------
# Otsu thresholding (segmentation)
# ---------------------------------------------------------------------------

def otsu_threshold(img: np.ndarray) -> int:
    """
    Find the optimal threshold by minimising intra-class variance (Otsu's method).

    Returns the integer threshold value t* in [0, 255].
    """
    pixels = img.flatten().astype(np.float64)
    total = len(pixels)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)

    best_t = 0
    best_var = np.inf

    w0, sum0 = 0.0, 0.0
    total_mean = np.sum(np.arange(256) * hist) / total

    for t in range(256):
        w0 += hist[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue

        sum0 += t * hist[t]
        mu0 = sum0 / w0
        mu1 = (total * total_mean - sum0) / w1

        # Intra-class variance = weighted sum of per-class variances
        # Equivalent to minimising: w0*w1*(mu0 - mu1)^2  (inter-class, same optimum)
        between = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        intra = -between  # maximise between ↔ minimise intra

        if intra < best_var:
            best_var = intra
            best_t = t

    return best_t


def segment_tire(img: np.ndarray) -> np.ndarray:
    """
    Binarise a grayscale image using the manual Otsu threshold.
    Returns a uint8 binary mask (0 or 255).
    """
    t = otsu_threshold(img)
    return (img >= t).astype(np.uint8) * 255


def defect_area_ratio(mask: np.ndarray) -> float:
    """
    Feature: fraction of foreground (white) pixels in the binary mask.
    """
    return float((mask > 0).sum() / mask.size)
