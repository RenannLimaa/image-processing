import numpy as np

def rect_kernel(size: int = 3) -> np.ndarray:
    return np.ones((size, size), dtype=np.uint8)


def cross_kernel(size: int = 3) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    k[mid, :] = 1
    k[:, mid] = 1
    return k



def erode(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
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
            if (region[kernel == 1]).min() == 1:
                out[i, j] = 1

    return (out * 255).astype(np.uint8)


def dilate(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
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
    return dilate(erode(img, kernel), kernel)


def closing(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    return erode(dilate(img, kernel), kernel)

def connected_regions_count(mask: np.ndarray) -> int:
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
