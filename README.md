# Tire Quality Classification

Binary classification: **Good** vs **Defective** tires using classical image processing and deep learning.

**Dataset:** [Tyre Quality Classification – Kaggle](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification)

---

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Download Dataset

```bash
uv run kaggle datasets download -d warcoder/tyre-quality-classification -p data/
unzip data/tyre-quality-classification.zip -d "data/tyre_dataset/Digital images of defective and good condition tyres"
```

(Requires `kaggle` CLI configured with API credentials.)

---

## What It Does

The notebook runs two parallel pipelines:

| Pipeline | Method | Accuracy | F1-score |
|---|---|---:|---:|
| Classical | SVM (RBF) on 9 hand-crafted features | 80.6% | 0.8302 |
| CNN | MobileNetV2 transfer learning | 90.1% | 0.9133 |

### Classical Pipeline (Sections 1–8)
- Manual preprocessing: grayscale, Gaussian blur, histogram equalization
- Manual edge detection (Canny) and segmentation (Otsu)
- Manual morphology (erosion, dilation, closing)
- 9-feature vector: frequency energy, entropy, statistics, edge density, connected regions
- ML models: SVM, KNN, Logistic Regression

### CNN Pipeline (Section 9)
- Frozen MobileNetV2 backbone (ImageNet pre-trained)
- RGB input 224×224, 15 epochs, Adam optimizer
- Checkpoint saved to `results/metrics/cnn_best.pt`

---

## Results

| File | Contents |
|---|---|
| `results/metrics/model_comparison_full.csv` | All 4 models side-by-side |
| `results/metrics/class_statistics.csv` | Feature means per class |
| `results/figures/` | 10 PNG visualizations |

---

## Key Files

- `src/preprocessing.py` — manual Gaussian blur, histogram equalization
- `src/edges.py` — manual Canny edge detection
- `src/features.py` — 9-feature assembly, dataset loader
- `src/cnn.py` — MobileNetV2 model, training loop
- `notebooks/main_pipeline.ipynb` — complete pipeline (Sections 1–9)

---

## Notes

- First run computes all 9 features (~10 min); cached in `results/metrics/features.csv`
- CNN training on CPU takes ~20–40 min
- Set `n_epochs=2` in Section 9 for a quick test
- Dataset is gitignored; download required each run
