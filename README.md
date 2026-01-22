# Brain Tumor Segmentation & Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

> **Note:** Original research notebooks preserved in [`archive/original-notebooks`](../../tree/archive/original-notebooks) branch.

## 🎯 Project Overview

End-to-end PyTorch pipeline for brain tumor segmentation and classification from MRI scans.

**Architecture:** Raw MRI → Preprocessing → U-Net Segmentation → Feature Extraction → Classification

## 📊 Target Performance

| Task | Model | Metric | Target |
|------|-------|--------|--------|
| Segmentation | U-Net | Dice | 92%+ |
| Segmentation | U-Net | IoU | 90%+ |
| Classification | Gradient Boosting | F1 | 94%+ |

## 🏗️ Tech Stack

- **Deep Learning:** PyTorch 2.0
- **Preprocessing:** OpenCV, scikit-image
- **Experiment Tracking:** MLflow
- **Testing:** pytest with coverage
- **CI/CD:** GitHub Actions (planned)

## 📁 Project Structure

```
brain-tumor-segmentation/
├── src/
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # PyTorch U-Net, metrics, losses
│   ├── features/          # Feature extraction (LBP, Gabor, GLCM)
│   ├── classification/    # Gradient Boosting classifier
│   ├── training/          # Training utilities
│   └── inference/         # End-to-end pipeline
├── experiments/           # MLflow training scripts
├── tests/                # Unit tests (pytest)
└── notebooks/            # Demo notebooks

Original research: archive/original-notebooks branch
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Install
pip install -r requirements.txt

# Train (coming soon)
python experiments/train_segmentation.py
```

## 🧪 Development Status

**Current Phase:** Week 1/3 - Core refactoring  
**Next Milestone:** Complete U-Net implementation + MLflow integration

### Roadmap

- [ ] Preprocessing pipeline (Wiener, CLAHE, cropping)
- [ ] PyTorch U-Net architecture
- [ ] Custom metrics (Dice, IoU) & losses
- [ ] Training script with MLflow
- [ ] Feature extraction module
- [ ] Classification pipeline
- [ ] Unit tests + CI/CD
- [ ] End-to-end inference pipeline

## 📚 Academic Context

This project evolved from a Pattern Recognition course final project (Dec 2025 - Jan 2026).

**Refactoring demonstrates:**
- Research-to-production code transformation
- ML engineering best practices
- Experiment tracking and reproducibility
- Comprehensive testing and CI/CD

**Original academic notebooks:** [`archive/original-notebooks`](../../tree/archive/original-notebooks)

## 📄 License

MIT

---

**Last Updated:** January 22, 2026 
**Status:** 🚧 Active refactoring in progress
