import numpy as np
from src.preprocessing import _convolve2d

SOBEL_GX = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)

SOBEL_GY = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float64)


def sobel_gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = _convolve2d(img.astype(np.float64), SOBEL_GX)
    gy = _convolve2d(img.astype(np.float64), SOBEL_GY)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.degrees(np.arctan2(gy, gx)) % 180 
    return magnitude, direction

def non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = direction[i, j]
            m = magnitude[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= angle < 67.5:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            elif 67.5 <= angle < 112.5:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]

            if m >= n1 and m >= n2:
                suppressed[i, j] = m

    return suppressed


def hysteresis(img: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15) -> np.ndarray:
    max_val = img.max()
    if max_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    high = high_ratio * max_val
    low = low_ratio * max_val

    strong = (img >= high).astype(np.uint8)
    weak = ((img >= low) & (img < high)).astype(np.uint8)

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
    magnitude, direction = sobel_gradient(img.astype(np.float64))
    suppressed = non_max_suppression(magnitude, direction)
    edges = hysteresis(suppressed, low_ratio=low_ratio, high_ratio=high_ratio)
    return edges


def edge_density(edge_map: np.ndarray) -> float:
    return float((edge_map > 0).sum() / edge_map.size)


def otsu_threshold(img: np.ndarray) -> int:
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

        between = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        intra = -between  

        if intra < best_var:
            best_var = intra
            best_t = t

    return best_t


def segment_tire(img: np.ndarray) -> np.ndarray:
    t = otsu_threshold(img)
    return (img >= t).astype(np.uint8) * 255


def defect_area_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).sum() / mask.size)
