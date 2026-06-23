import numpy as np

def dct_1d(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    result = np.zeros(n, dtype=np.float64)
    ns = np.arange(n, dtype=np.float64)
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        result[k] = alpha * np.sum(signal * np.cos(np.pi * (2 * ns + 1) * k / (2 * n)))
    return result


def dct_2d(block: np.ndarray) -> np.ndarray:
    tmp = np.apply_along_axis(dct_1d, axis=1, arr=block.astype(np.float64))
    return np.apply_along_axis(dct_1d, axis=0, arr=tmp)

def block_dct(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    img = img.astype(np.float64)
    h, w = img.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    img = img[:h_crop, :w_crop]

    out = np.zeros_like(img)
    for row in range(0, h_crop, block_size):
        for col in range(0, w_crop, block_size):
            block = img[row : row + block_size, col : col + block_size]
            out[row : row + block_size, col : col + block_size] = dct_2d(block)
    return out


def freq_energy_ratio(
    dct_image: np.ndarray, block_size: int = 8, low_k: int = 2
) -> float:
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
