# MACS-30100: Perspectives on Computational Modeling

Welcome! This repository contains course projects and materials developed during my graduate coursework in computational modeling and machine learning at the University of Chicago.

## Highlighted Project: Deepfake Detection Pipeline (P3)

The `P3` folder contains my first year capstone project from this course, which focuses on the detection of deepfake videos using both deep learning and machine learning methods. The goal was to build, evaluate, and compare the performance of models trained on frame-level data extracted from deepfake videos.

**Project Directory**: [`P3: Deepfake Detection using Deep Learning and Machine Learning`](./P3%3A%20Deepfake%20Detection%20using%20Deep%20Learning%20and%20Machine%20Learning/)

### 🔍 Project Overview

This project implements two approaches to classify deepfake content:

- **Deep Learning Model**: EfficientNet-based CNN for feature extraction and classification  
- **Machine Learning Model**: Random Forest classifier trained on image-based features  

Performance was measured using standard metrics: Precision, Recall, F1-score, Confusion Matrix, and ROC/PR curves.

### Repository Structure (within `P3/`)
| File/Folder                | Description |
|---------------------------|-------------|
| `DF_Detection_Models.ipynb` | End-to-end pipeline: data prep, training, evaluation, and visualization |
| `JonSlidesP3.pdf`         | Final presentation deck summarizing the project |
| `README.md`               | Project documentation |
| `Models/`                 | Saved model files (`.pth`, `.joblib`) |
| `Scripts/`                | Scripts for dataset loading and frame extraction |
| `Data/ (external)`        | Dataset is stored externally via Google Drive due to size constraints |

**[Dataset Source (Kaggle)](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)**

### ⚙️ Model Summaries

#### EfficientNet (Deep Learning)

- **Why**: Optimized for image classification with fewer parameters and better accuracy.
- **F1-score**: 0.9708  
- **Precision**: 0.9826  
- **Recall**: 0.9594  

#### Random Forest (Machine Learning)

- **Why**: Strong performance on high-dimensional structured data; robust to noise and overfitting.
- **F1-score**: 0.9654  
- **Precision**: 0.9332  
- **Recall**: 0.9999  
- **ROC AUC**: 0.63

### Key Takeaways

- EfficientNet had stronger overall balance between precision and recall.
- Random Forest achieved high recall for fake frames but struggled with false positives.
- Deepfake detection benefits from combining modalities and model types.

### Future Directions

- Incorporate audio and metadata for multimodal learning  
- Experiment with ensemble models  
- Address class imbalance via targeted sampling or loss reweighting

---

## Contact

If you have questions or would like to discuss this work further, feel free to reach out:

**Name**: Jonathan Monroe  
**Email**: [jonathanmonroe@uchicago.edu](mailto:jonathanmonroe@uchicago.edu)  
**GitHub**: [github.com/MACS-30100-2025winter/macs-30100-JonathanPMonroe](https://github.com/MACS-30100-2025winter/macs-30100-JonathanPMonroe.git)
