# Brain Tumor Detection and Classification Using Deep Learning

> **Pattern Recognition Course Project | Universitas Gadjah Mada | June 2025**

[![View on Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebooks-20BEFF?logo=kaggle)](https://www.kaggle.com/bambangabhinawap/notebooks)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Project Overview

This project implements a comprehensive brain tumor detection and classification system comparing **U-Net deep learning segmentation** versus **traditional hybrid entropy-MultiOtsu thresholding** approaches, combined with traditional machine learning classifiers for tumor classification.

This work was completed as a **group project** for the Pattern Recognition course at Universitas Gadjah Mada, with a complete technical report documented in IEEE format.

### 🎯 Key Achievements

- **92.05% F1-score** achieved using Logistic Regression with U-Net segmentation features
- **6% performance improvement** of U-Net over traditional thresholding methods
- Successfully processed and analyzed **1300+ MRI brain scans** across 4 tumor classes
- Implemented advanced feature extraction: **LBP, Gabor Wavelets, and GLCM**
- **90% weighted recall** indicating low false negative risk for medical diagnosis

### 👥 Project Team

**Group 3 - Department of Computer Science and Electronics, Universitas Gadjah Mada**

- Bambang Abhinawa Pinakasakti
- Salwaa Mumtaazah Darmanastri
- Salman Faiz Hidayat
- Raden Rara Garzetta Aleyda Harimurti
- Adam Maulana Haq
- Puti Khayira Naila

---

## 📊 Project Notebooks

This repository currently includes two segmentation notebooks—one deep learning (U‑Net) and one traditional thresholding approach.
The additional notebooks for U‑Net mask prediction and classification (both U‑Net and thresholding) are available on Kaggle.

### Segmentation Pipeline

**1. [U-Net Model Training](https://www.kaggle.com/code/bambangabhinawap/trainingunetbraintumor)**
   - Train deep learning U-Net architecture for tumor segmentation
   - Encoder-decoder structure with skip connections
   - Evaluated using Dice coefficient and IoU metrics
   - Trained on external dataset with human-annotated masks

**2. [U-Net Mask Prediction](https://www.kaggle.com/code/bambangabhinawap/u-net-mask-prediction-brain-tumor)**
   - Generate segmentation masks on primary dataset using trained U-Net
   - Save probability maps with 0.5 threshold for binary masks
   - Outputs used for downstream feature extraction

**3. [Threshold-Based Segmentation](https://www.kaggle.com/code/bambangabhinawap/braintumorthresholding)**
   - Hybrid entropy-MultiOtsu thresholding method
   - Local entropy with 5-pixel disk kernel (threshold: 0.5625)
   - Morphological operations (erosion + dilation) for mask refinement
   - Baseline comparison approach

### Classification Pipeline

**4. [Classification with U-Net Features](https://www.kaggle.com/code/bambangabhinawap/pp-finalproject-braintumordetection)**
   - Extract LBP, Gabor, and GLCM features from U-Net segmented regions
   - Train SVM, Logistic Regression, Random Forest, and Gradient Boosting classifiers
   - **Best performance: 92.05% F1-score with Logistic Regression**

**5. [Classification with Threshold Features](https://www.kaggle.com/code/bambangabhinawap/pp-finalproject-braintumordetection-threshold)**
   - Extract features from threshold-segmented regions
   - Same classifier comparison with traditional segmentation
   - **Best performance: 86.23% F1-score with Random Forest**

---

## 📄 Technical Report

A complete **IEEE-format technical report** documenting this project is available in this repository:

**[View Technical Report (PDF)](Group%203%20-%20Final%20Report%20-%20Pattern%20Recognition%20Final%20Project.pdf)**

The report includes:
- Comprehensive literature review
- Detailed methodology for each pipeline stage
- Experimental results with confusion matrices
- Statistical analysis and model comparisons
- References to 30+ academic papers

---

## 🔬 Technical Methodology

### 1. Image Preprocessing Pipeline

**Enhancement**
- **Wiener Filtering**: Noise suppression while preserving tumor edge information
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Localized contrast enhancement with clip limit 2.0
  - Splits image into tiles for adaptive histogram equalization
  - Prevents artificial amplification of homogeneous regions

**Normalization**
- Pixel intensity scaling to [0, 1] range by dividing by 255.0
- Ensures consistent data distribution across different MRI scanners
- Improves training stability and convergence

**Cropping**
- Automated brain region extraction using contour detection
- Gaussian blurring (3×3 kernel) followed by thresholding (45/255)
- Morphological operations (erosion + dilation) to refine mask
- Standardizes input dimensions and focuses computation on relevant regions

**Augmentation**
- **Horizontal flipping**: Exploits bilateral symmetry of brain hemispheres
- **Brightness adjustment**: Simulates scanner variations and patient-specific factors
- Improves model robustness to real-world clinical variability

### 2. Segmentation Approaches

| Method | Technique | Architecture/Algorithm | F1-Score Impact |
|--------|-----------|----------------------|-----------------|
| **U-Net** | Deep Learning | Encoder-decoder with skip connections | ⭐ **92% (+6%)** |
| **Hybrid Thresholding** | Traditional CV | Entropy + MultiOtsu + Morphology | **86% (baseline)** |

**U-Net Architecture:**
- Pretrained on external dataset with human annotations
- Combined loss: Dice + Binary Cross-Entropy
- Handles class imbalance effectively
- Best for diffuse or low-contrast tumors

**Hybrid Entropy-MultiOtsu:**
- Local entropy < 0.5625 set to black, higher to white
- MultiOtsu creates 3 intensity classes, highest marked as tumor
- Bitwise AND operation for final mask
- Best for high-contrast, well-defined tumors

### 3. Feature Extraction

Advanced texture and spatial feature extraction from segmented tumor regions:

**Local Binary Pattern (LBP)**
- **Parameters**: P=8 neighbors, R=1 radius, uniform patterns
- **Output**: 10-dimensional feature vector (L1 normalized histogram)
- **Captures**: Local texture variations from cellular growth changes
- Compares each pixel with 8 circular neighbors

**Gabor Wavelet Transform (GWT)**
- **Parameters**: 2 frequencies (0.1, 0.3) × 4 orientations (0°, 45°, 90°, 135°)
- **Output**: 16-dimensional feature vector (mean + variance per filter)
- **Captures**: Multi-scale and orientation-specific texture patterns
- Effective for detecting irregular tumor boundaries

**Gray-Level Co-occurrence Matrix (GLCM)**
- **Parameters**: Distance=1, 4 angles (0°, 45°, 90°, 135°), 256 gray levels
- **Output**: 24-dimensional feature vector (6 properties × 4 directions)
- **Properties**: Contrast, dissimilarity, homogeneity, ASM, energy, correlation
- **Captures**: Second-order spatial texture relationships
- Preserves directional sensitivity for better classification

**Total Feature Dimension**: 50 features (10 LBP + 16 GWT + 24 GLCM)

### 4. Classification & Evaluation

**Models Tested:**
- Support Vector Machine (SVM)
- Logistic Regression ⭐
- Random Forest
- Gradient Boosting
- Voting Ensemble

**Evaluation Metrics:**
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate (critical for medical diagnosis)
- **F1-Score**: Harmonic mean of precision and recall

---

## 📈 Performance Comparison

### Classification Results with U-Net Segmentation

| Classifier | F1-Score | Notes |
|------------|----------|-------|
| **Logistic Regression** | **92.05%** ✨ | Best overall - efficient and interpretable |
| SVM | 90% | Strong performance, high-dimensional data |
| Random Forest | 90% | Competitive ensemble method |
| Gradient Boosting | 89% | Good but slightly overfits |
| Voting Ensemble | 92.03% | Combined predictions, minimal improvement |

**Key Insight**: Logistic Regression performs best because extracted features (LBP + GWT + GLCM) are sufficiently informative and likely linearly separable.

### Classification Results with Threshold Segmentation

| Classifier | F1-Score | Notes |
|------------|----------|-------|
| Logistic Regression | 84% | Less effective with noisy features |
| **Random Forest** | **86.23%** ✨ | Best for handling non-linear patterns |
| Gradient Boosting | 84% | Similar to logistic regression |
| Voting Ensemble | 85.79% | Moderate improvement |

**Key Insight**: Random Forest performs best with thresholding because it handles noisy, non-linear features better through ensemble of decision trees.

### Segmentation Quality Comparison

**U-Net Advantages:**
- Focused segmentation tightly around tumor boundaries
- Learned features from diverse training data
- Handles low-contrast and diffuse tumors
- Consistent mask quality across cases

**Thresholding Limitations:**
- Inconsistent segmentation quality
- Sometimes segments skull regions instead of tumor
- Struggles with low-contrast lesions
- Sensitive to intensity variations

---

## 🛠️ Technologies & Tools

**Deep Learning & Neural Networks**
- TensorFlow 2.13
- Keras 2.13

**Computer Vision & Image Processing**
- OpenCV 4.8
- scikit-image 0.21
- PIL/Pillow 10.0

**Machine Learning**
- scikit-learn 1.3 (SVM, Logistic Regression, Random Forest, Gradient Boosting)
- scipy 1.11

**Feature Engineering**
- Local Binary Pattern (LBP) - skimage.feature
- Gabor Wavelet Transform - skimage.filters
- GLCM texture features - skimage.feature

**Data Processing & Visualization**
- NumPy 1.24
- Pandas 2.0
- Matplotlib 3.7
- seaborn 0.12

---

## 📈 Dataset Information

**Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- **Training Set**: ~2,870 MRI images
- **Testing Set**: ~394 MRI images
- **Total**: ~3,264 grayscale MRI scans
- **Image Classes** (4):
  - No Tumor
  - Glioma (malignant)
  - Meningioma (usually benign)
  - Pituitary Tumor
- **Format**: Grayscale MRI scans, resized to 128×128 pixels
- **Preprocessing**: Wiener filtering, CLAHE, normalization, augmentation

---

## 🚀 How to Reproduce

### Option 1: Run on Kaggle (Recommended)

1. Visit any notebook link above
2. Click **"Copy & Edit"** to fork the notebook to your Kaggle account
3. Enable GPU acceleration: **Settings → Accelerator → GPU**
4. Run all cells sequentially
5. Notebooks run in order: Training → Prediction → Segmentation → Classification

**Advantages:**
- Pre-installed dependencies
- Free GPU access
- Dataset readily available
- No local setup required

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# Extract to data/ directory

# Run notebooks in order
jupyter notebook notebooks/
```

**Requirements:**
- Python 3.8+
- CUDA-compatible GPU (recommended for U-Net training)
- 8GB+ RAM
- ~5GB disk space for dataset

---

## 📁 Repository Structure

```
brain-tumor-detection/
├── notebooks/
│   ├── trainingunetbraintumor.ipynb      # U‑Net training notebook
│   └── braintumorthresholding.ipynb      # Threshold segmentation notebook
├── Group‑3‑Final‑Report‑Pattern‑Recognition‑Final‑Project.pdf
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

```

---

## 🎓 Academic Context

This project was developed as the **final group project** for the **Pattern Recognition** course at **Universitas Gadjah Mada** in June 2025. The project explores:

- Comparative analysis of deep learning vs traditional computer vision techniques
- Impact of segmentation quality on downstream classification tasks
- Feature engineering for medical image analysis
- Multi-class tumor classification challenges in medical imaging
- Interpretability vs performance trade-offs in ML models

**Course**: Pattern Recognition  
**Institution**: Department of Computer Science and Electronics, Universitas Gadjah Mada  
**Semester**: June 2025  
**Grade**: [To be added]

---

## 📊 Key Findings

1. **U-Net significantly outperforms traditional thresholding** (+6% F1-score improvement)
   - Better segmentation quality → better features → better classification

2. **Segmentation method affects optimal classifier choice**
   - U-Net: Logistic Regression (92.05%)
   - Thresholding: Random Forest (86.23%)

3. **Feature extraction is critical**
   - Combined LBP + GWT + GLCM captures complementary texture information
   - 50-dimensional feature space provides sufficient discriminative power

4. **Classical ML remains competitive**
   - With proper feature engineering, traditional ML achieves >92% F1-score
   - More interpretable than end-to-end deep learning
   - Suitable for limited data scenarios

5. **Medical diagnosis considerations**
   - High recall (90%) minimizes false negatives
   - Critical for early tumor detection
   - Balance between sensitivity and specificity achieved

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Implement attention mechanisms in U-Net architecture
- [ ] Explore transformer-based segmentation (Swin-UNET, TransUNet)
- [ ] Add explainability with Grad-CAM visualizations
- [ ] Experiment with ensemble U-Net models

### Dataset & Features
- [ ] Extend to 3D volumetric MRI analysis
- [ ] Incorporate multi-modal MRI (T1, T2, FLAIR)
- [ ] Add radiomics features
- [ ] Implement deep feature extraction (CNN embeddings)

### Clinical Application
- [ ] Multi-class subtype classification (glioma grades)
- [ ] Tumor size estimation and progression tracking
- [ ] Deploy as web application for clinical demonstration
- [ ] Integration with DICOM medical image format

### Performance
- [ ] Hyperparameter optimization (GridSearchCV, Bayesian)
- [ ] Cross-validation for robust evaluation
- [ ] Test on external datasets for generalization

---

## 👤 Author (Repository Maintainer)

**Bambang Abhinawa Pinakasakti**  
Computer Science Student | University of Birmingham (Dual Degree with UGM)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/bambang-abhinawa-pinakasakti-6955092b4/) [![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle)](https://www.kaggle.com/bambangabhinawap)

*This project was completed as part of a 6-person team at Universitas Gadjah Mada. Full team member list available in the technical report.*

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

This is an academic project for educational purposes. Dataset usage follows Kaggle dataset licensing terms.

---

## 🙏 Acknowledgments

- **Universitas Gadjah Mada** - Pattern Recognition Course instructors and support
- **Kaggle** - Brain Tumor MRI Dataset by Masoud Nickparvar
- **TensorFlow & scikit-learn communities** - Open-source tools and documentation
- **Group 3 team members** - Collaborated in the applied research, experimental design, and documentation process of this study.

---

## 📚 References

The complete reference list (30+ academic papers) is available in the [technical report PDF](Group-3-Final-Report-Pattern-Recognition-Final-Project.pdf).

Key references include:
- Ojala et al. (1996) - Local Binary Pattern methodology
- Haralick et al. (1973) - GLCM texture features
- Bovik et al. (1990) - Gabor wavelet transforms
- Recent works on U-Net segmentation for medical imaging
- Brain tumor classification using traditional ML and deep learning

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

*All notebooks with complete outputs, visualizations, and interactive code available on [Kaggle](https://www.kaggle.com/bambangabhinawap/notebooks)*

**[View Technical Report](Group-3-Final-Report-Pattern-Recognition-Final-Project.pdf)** | **[View Notebooks on Kaggle](https://www.kaggle.com/bambangabhinawap/notebooks)**

</div>
