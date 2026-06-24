# Tire Quality Classification — Image Processing Final Project

Binary classification of tire images as **good** or **defective** using a hand-crafted image processing pipeline.

## Problem

Worn or damaged tires are a leading cause of road accidents. This project applies classical image processing and machine learning techniques to automatically classify tire images from visual inspection data.

**Dataset:** [Tyre Quality Classification — Kaggle](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification)  
1,854 digital images split into two classes: `good` and `defective`.

---

## Techniques Used

| Stage | Technique | Implementation |
|---|---|---|
| Pre-processing | Grayscale conversion (luminosity weights) | Manual |
| Pre-processing | Gaussian blur (2-D convolution) | Manual |
| Pre-processing | Histogram equalisation (CDF mapping) | Manual |
| Frequency domain | Block DCT-II (8×8 blocks) | Manual |
| Descriptive analysis | Shannon entropy | Manual |
| Descriptive analysis | Mean, Std, Skewness, Kurtosis | Manual |
| Edge detection | Canny (Sobel → NMS → Hysteresis) | Manual |
| Segmentation | Otsu thresholding | Manual |
| Morphology | Erosion, Dilation, Opening, Closing | Manual |
| Machine learning | SVM (RBF), KNN, Logistic Regression | scikit-learn |

---

## Project Structure

```
image-processing/
├── data/tyre_dataset/         # Place dataset here
│   ├── defective/
│   └── good/
├── notebooks/
│   └── main_pipeline.ipynb    # Main notebook (all 8 sections)
├── src/
│   ├── preprocessing.py       # Grayscale, Gaussian blur, histogram equalisation
│   ├── frequency.py           # Manual DCT-II and block DCT
│   ├── descriptors.py         # Entropy, image statistics
│   ├── edges.py               # Manual Canny, Otsu thresholding
│   ├── morphology.py          # Erosion, dilation, opening, closing
│   └── features.py            # Feature vector assembly and dataset loader
├── results/
│   ├── figures/               # Saved plots
│   └── metrics/               # CSV results (features, model comparison)
└── pyproject.toml
```

---

## Setup and Execution

### 1. Install dependencies

```bash
uv sync
```

### 2. Download the dataset

Log in to Kaggle and place your `kaggle.json` API key at `~/.config/kaggle/kaggle.json`, then:

```bash
uv run kaggle datasets download warcoder/tyre-quality-classification -p data/
unzip data/tyre-quality-classification.zip -d data/tyre_dataset/
```

Alternatively, download manually from [Kaggle](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification) and extract to `data/tyre_dataset/` with `defective/` and `good/` sub-folders.

### 3. Run the notebook

```bash
uv run jupyter notebook notebooks/main_pipeline.ipynb
```

Execute cells in order. Feature extraction for all ~1,854 images is cached to `results/metrics/features.csv` after the first run.

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| SVM (RBF) | — | — | — | — |
| KNN (best k) | — | — | — | — |
| Logistic Regression | — | — | — | — |

*(Table is populated automatically by Section 7 of the notebook.)*

Figures saved to `results/figures/`. Detailed metrics saved to `results/metrics/`.
