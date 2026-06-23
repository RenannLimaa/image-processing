import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def save_image(path: str, img: np.ndarray) -> None:
    cv2.imwrite(path, img)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    b = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    r = img[:, :, 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)

def resize_image(img: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    k = kernel_size // 2
    y, x = np.mgrid[-k : k + 1, -k : k + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (padded[i : i + kh, j : j + kw] * kernel).sum()
    return out

def gaussian_blur(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    kernel = _gaussian_kernel(kernel_size, sigma)
    if img.ndim == 2:
        blurred = _convolve2d(img, kernel)
        return np.clip(blurred, 0, 255).astype(np.uint8)
    channels = [_convolve2d(img[:, :, c], kernel) for c in range(img.shape[2])]
    blurred = np.stack(channels, axis=-1)
    return np.clip(blurred, 0, 255).astype(np.uint8)

def histogram_equalize(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("histogram_equalize expects a single-channel grayscale image.")
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()

    cdf_min = cdf[cdf > 0].min()
    total_pixels = img.size
    lut = np.round(
        (cdf - cdf_min) / (total_pixels - cdf_min) * 255
    ).astype(np.uint8)
    return lut[img]

def normalize(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 255.0

def preprocess(path: str, size: tuple[int, int] = (256, 256)) -> dict:
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
