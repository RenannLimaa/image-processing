"""
Descriptors module — statistical and information-theoretic image descriptors.

All computations are manual (numpy only, no scipy.stats shortcuts for the
main feature functions).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Histogram utility
# ---------------------------------------------------------------------------

def pixel_histogram(img: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalised pixel intensity histogram.

    Returns
    -------
    probs : ndarray of shape (bins,)  — probability of each intensity level
    edges : ndarray of shape (bins+1,) — bin edges
    """
    counts, edges = np.histogram(img.flatten(), bins=bins, range=(0, 256))
    probs = counts / counts.sum()
    return probs, edges


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def entropy(img: np.ndarray) -> float:
    """
    Compute the Shannon entropy of the pixel intensity distribution:

        H = -sum_i  p_i * log2(p_i)    (summing only non-zero bins)

    Higher entropy → more disordered texture (typically defective tires).
    """
    probs, _ = pixel_histogram(img)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


# ---------------------------------------------------------------------------
# Higher-order statistics
# ---------------------------------------------------------------------------

def image_statistics(img: np.ndarray) -> dict:
    """
    Compute mean, std, skewness, and kurtosis of pixel intensities manually.

    Skewness  = E[(X - mu)^3] / sigma^3
    Kurtosis  = E[(X - mu)^4] / sigma^4   (excess kurtosis: subtract 3)

    Returns a dict with keys: mean, std, skewness, kurtosis.
    """
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
