import numpy as np


def pixel_histogram(img: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(img.flatten(), bins=bins, range=(0, 256))
    probs = counts / counts.sum()
    return probs, edges


def entropy(img: np.ndarray) -> float:
    probs, _ = pixel_histogram(img)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def image_statistics(img: np.ndarray) -> dict:
    pixels = img.flatten().astype(np.float64)
    n = len(pixels)
    mu = pixels.mean()
    sigma = pixels.std()

    if sigma == 0:
        return {"mean": float(mu), "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}

    deviations = pixels - mu
    skewness = float(np.sum(deviations**3) / (n * sigma**3))
    kurtosis = float(np.sum(deviations**4) / (n * sigma**4) - 3)  # excess kurtosis

    return {
        "mean": float(mu),
        "std": float(sigma),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
