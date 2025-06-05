# University of Chicago — First-Year Capstone Project  
**Jonathan Monroe**

This repository contains the materials for my first-year capstone project in computational modeling and machine learning at the University of Chicago.

## Project: Deepfake Detection Pipeline (P3)

The `P3` folder contains a capstone project focused on the detection of deepfake videos using both deep learning and machine learning methods. The objective was to build, evaluate, and compare the performance of models trained on frame-level image data.

**Project Directory**: `P3: Deepfake Detection using Deep Learning and Machine Learning`

## Project Overview

The project implements two classification approaches:

- Deep Learning: EfficientNet-based CNN for feature extraction and classification  
- Machine Learning: Random Forest classifier trained on image-derived features  

Performance was evaluated using standard metrics, including precision, recall, F1-score, confusion matrix, ROC AUC, and precision-recall curves.

## Repository Structure (within `P3/`)

| File/Folder              | Description |
|--------------------------|-------------|
| `DF_Detection_Models.ipynb` | Complete pipeline for data preprocessing, model training, evaluation, and visualization |
| `JonSlidesP3.pdf`        | Final presentation summarizing the project |
| `Models/`                | Trained model files (`.pth`, `.joblib`) |
| `Scripts/`               | Data preparation and frame extraction scripts |
| `Data/ (external)`       | Dataset hosted externally due to size constraints |

**Dataset Source**: [Kaggle Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)

## Model Summaries

### EfficientNet (Deep Learning)

- Designed for high-performance image classification
- F1-score: 0.9708  
- Precision: 0.9826  
- Recall: 0.9594  

### Random Forest (Machine Learning)

- Effective with high-dimensional feature spaces  
- F1-score: 0.9654  
- Precision: 0.9332  
- Recall: 0.9999  
- ROC AUC: 0.63

## Key Findings

- EfficientNet provided strong balance between precision and recall  
- Random Forest achieved high recall but produced a higher false positive rate  
- Combining deep learning and traditional machine learning can enhance detection robustness

## Future Work

- Incorporate additional modalities such as audio and metadata  
- Evaluate ensemble methods that combine DL and ML predictions  
- Apply sampling strategies and loss reweighting to address class imbalance

## Contact

For questions or further discussion:

- Name: Jonathan Monroe  
- Email: [jonathanmonroe@uchicago.edu](mailto:jonathanmonroe@uchicago.edu)  
- GitHub: [github.com/JonathanPMonroe/first-year-capstone-sample](https://github.com/JonathanPMonroe/first-year-capstone-sample)
