"""
Feature assembly module.

Orchestrates all processing stages to produce a fixed-length feature vector
for each image, then loads the entire dataset.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing import preprocess
from src.frequency import block_dct, freq_energy_ratio
from src.descriptors import entropy, image_statistics
from src.edges import canny, edge_density, segment_tire, defect_area_ratio
from src.morphology import closing, rect_kernel, connected_regions_count


FEATURE_NAMES = [
    "freq_energy_ratio",
    "entropy",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "edge_density",
    "defect_area_ratio",
    "connected_regions",
]


def build_feature_vector(img_path: str) -> np.ndarray:
    """
    Run the full processing pipeline on a single image and return a 9-element
    feature vector:

        [freq_energy_ratio, entropy, mean, std, skewness, kurtosis,
         edge_density, defect_area_ratio, connected_regions]
    """
    stages = preprocess(img_path)
    eq = stages["equalized"]  # histogram-equalised, blurred, grayscale 256×256

    # --- Phase 3: Frequency domain ---
    dct_img = block_dct(eq)
    f_freq = freq_energy_ratio(dct_img)

    # --- Phase 4: Descriptive analysis ---
    f_entropy = entropy(eq)
    stats = image_statistics(eq)
    f_mean = stats["mean"]
    f_std = stats["std"]
    f_skew = stats["skewness"]
    f_kurt = stats["kurtosis"]

    # --- Phase 5: Edges ---
    edges = canny(eq)
    f_edge = edge_density(edges)

    # --- Phase 5: Segmentation + morphology ---
    mask = segment_tire(eq)
    kernel = rect_kernel(3)
    mask_closed = closing(mask, kernel)
    f_defect = defect_area_ratio(mask_closed)
    f_regions = float(connected_regions_count(mask_closed))

    return np.array([
        f_freq, f_entropy, f_mean, f_std, f_skew, f_kurt,
        f_edge, f_defect, f_regions,
    ], dtype=np.float64)


def load_dataset(
    data_dir: str | Path,
    save_csv: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Iterate all images in ``data_dir`` (expects sub-folders named 'defective'
    and 'good') and compute feature vectors for every image.

    Parameters
    ----------
    data_dir : path to the dataset root folder
    save_csv : if given, save the feature matrix + labels to this CSV path

    Returns
    -------
    X       : float64 array of shape (N, 9)
    y       : int array of shape (N,)  — 0=good, 1=defective
    paths   : list of image file paths (length N)
    """
    data_dir = Path(data_dir)
    class_map = {"good": 0, "defective": 1}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    all_paths: list[str] = []
    all_labels: list[int] = []

    for class_name, label in class_map.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        files = [
            p for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in extensions
        ]
        all_paths.extend(str(p) for p in files)
        all_labels.extend([label] * len(files))

    X_rows = []
    failed = []
    for path in tqdm(all_paths, desc="Extracting features"):
        try:
            X_rows.append(build_feature_vector(path))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skipping {path}: {exc}")
            failed.append(path)
            # Remove corresponding label
            idx = all_paths.index(path)
            all_labels.pop(idx)
            all_paths.remove(path)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(all_labels, dtype=np.int32)

    if save_csv is not None:
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df.insert(0, "path", all_paths)
        df.insert(1, "label", y)
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Feature matrix saved to {save_csv}")

    return X, y, all_paths
