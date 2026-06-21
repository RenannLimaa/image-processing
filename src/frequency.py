"""
Frequency domain module — manual block DCT implementation.

No scipy.fft or numpy.fft is used for the main transforms.
The DCT-II formula is applied manually on 8×8 image blocks.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Manual DCT-II (1-D)
# ---------------------------------------------------------------------------

def dct_1d(signal: np.ndarray) -> np.ndarray:
    """
    Compute the DCT-II of a 1-D signal of length N:

        X[k] = alpha(k) * sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k / (2N))

    where alpha(0) = sqrt(1/N), alpha(k>0) = sqrt(2/N).
    """
    n = len(signal)
    result = np.zeros(n, dtype=np.float64)
    ns = np.arange(n, dtype=np.float64)
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        result[k] = alpha * np.sum(
            signal * np.cos(np.pi * (2 * ns + 1) * k / (2 * n))
        )
    return result


def dct_2d(block: np.ndarray) -> np.ndarray:
    """
    Compute the 2-D DCT-II by applying 1-D DCT to each row then each column.
    Input: square 2-D float64 array (typically 8×8).
    """
    tmp = np.apply_along_axis(dct_1d, axis=1, arr=block.astype(np.float64))
    return np.apply_along_axis(dct_1d, axis=0, arr=tmp)


# ---------------------------------------------------------------------------
# Block DCT over a full image
# ---------------------------------------------------------------------------

def block_dct(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Divide a grayscale image into non-overlapping `block_size × block_size`
    blocks and apply dct_2d to each block.

    The image is cropped to the nearest multiple of block_size in each
    dimension before processing.

    Returns an array of the same (cropped) shape containing DCT coefficients.
    """
    img = img.astype(np.float64)
    h, w = img.shape
    # Crop to multiples of block_size
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    img = img[:h_crop, :w_crop]

    out = np.zeros_like(img)
    for row in range(0, h_crop, block_size):
        for col in range(0, w_crop, block_size):
            block = img[row : row + block_size, col : col + block_size]
            out[row : row + block_size, col : col + block_size] = dct_2d(block)
    return out


# ---------------------------------------------------------------------------
# Feature: frequency energy ratio
# ---------------------------------------------------------------------------

def freq_energy_ratio(dct_image: np.ndarray, block_size: int = 8, low_k: int = 2) -> float:
    """
    Compute the ratio of low-frequency energy to total energy across all DCT blocks.

    For each 8×8 block the low-frequency energy is the squared sum of the
    top-left `low_k × low_k` coefficients; the total energy is the squared
    sum of all coefficients.

    A value close to 1 → energy concentrated in low frequencies (smooth texture).
    A lower value → more high-frequency content (irregular / defective surface).
    """
    h, w = dct_image.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    dct_image = dct_image[:h_crop, :w_crop]

    low_energy = 0.0
    total_energy = 0.0
    for row in range(0, h_crop, block_size):
        for col in range(0, w_crop, block_size):
            block = dct_image[row : row + block_size, col : col + block_size]
            block_sq = block**2
            total_energy += block_sq.sum()
            low_energy += block_sq[:low_k, :low_k].sum()

    if total_energy == 0:
        return 1.0
    return float(low_energy / total_energy)
