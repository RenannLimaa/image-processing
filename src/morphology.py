"""
Morphology module — manual erosion, dilation, opening, and closing.

No cv2 morphological functions are used.
All operations use a sliding-window (min/max) approach.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Structuring element helpers
# ---------------------------------------------------------------------------

def rect_kernel(size: int = 3) -> np.ndarray:
    """Return a square flat (all-ones) structuring element."""
    return np.ones((size, size), dtype=np.uint8)


def cross_kernel(size: int = 3) -> np.ndarray:
    """Return a cross-shaped structuring element."""
    k = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    k[mid, :] = 1
    k[:, mid] = 1
    return k


# ---------------------------------------------------------------------------
# Core morphological operations
# ---------------------------------------------------------------------------

def erode(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Binary erosion: a pixel is 1 only if ALL pixels under the kernel are 1.

    For binary images (0 / 255) this is equivalent to a minimum filter.
    """
    if kernel is None:
        kernel = rect_kernel(3)

    binary = (img > 0).astype(np.uint8)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    out = np.zeros_like(binary)

    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            # Erosion: minimum over the structuring element positions
            if (region[kernel == 1]).min() == 1:
                out[i, j] = 1

    return (out * 255).astype(np.uint8)


def dilate(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Binary dilation: a pixel is 1 if ANY pixel under the kernel is 1.

    Equivalent to a maximum filter.
    """
    if kernel is None:
        kernel = rect_kernel(3)

    binary = (img > 0).astype(np.uint8)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    out = np.zeros_like(binary)

    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            if (region[kernel == 1]).max() == 1:
                out[i, j] = 1

    return (out * 255).astype(np.uint8)


def opening(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Morphological opening = erosion then dilation.
    Removes small noise (isolated bright pixels).
    """
    return dilate(erode(img, kernel), kernel)


def closing(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Morphological closing = dilation then erosion.
    Fills small holes / connects fragmented crack regions.
    """
    return erode(dilate(img, kernel), kernel)


# ---------------------------------------------------------------------------
# Feature: connected region count (simple flood-fill labelling)
# ---------------------------------------------------------------------------

def connected_regions_count(mask: np.ndarray) -> int:
    """
    Count the number of connected foreground components in a binary mask
    using iterative flood-fill (4-connectivity).

    More connected regions → more fragmented defects → tends to correlate
    with defective tires.
    """
    binary = (mask > 0).astype(np.int32)
    visited = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape
    count = 0

    for start_i in range(h):
        for start_j in range(w):
            if binary[start_i, start_j] == 1 and not visited[start_i, start_j]:
                # BFS flood-fill
                count += 1
                stack = [(start_i, start_j)]
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or ci >= h or cj < 0 or cj >= w:
                        continue
                    if visited[ci, cj] or binary[ci, cj] == 0:
                        continue
                    visited[ci, cj] = True
                    stack.extend([(ci + 1, cj), (ci - 1, cj),
                                   (ci, cj + 1), (ci, cj - 1)])

    return count
