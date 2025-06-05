# **Deepfake Detection Using Deep Learning and Machine Learning**

## **Project Overview**
This project explores the detection of deepfake videos by applying **both deep learning (DL) and machine learning (ML) models** to frame-level image data. The goal is to compare model performance and determine an effective approach for deepfake detection.

- **Deep Learning Model:** Convolutional Neural Network trained using EfficientNet for feature extraction and classification  
- **Machine Learning Model:** Random Forest for classification based on extracted image features  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC & Precision-Recall Curves  

---

## **Repository Structure**

### 📂 Root Directory
| File/Folder | Description |
|-------------------------|------------------------------------------------------------|
| `DF_Detection_Models.ipynb` | Jupyter Notebook containing the full pipeline for deepfake detection, including data preprocessing, model training (both DL & ML), evaluation, and visualization. |
| `JonSlidesP3.pdf` | Final presentation slides summarizing the project. |
| `README.md` | Project documentation, methodology, and results overview. |

### 📂 Models/
| File | Description |
|-------------------------|------------------------------------------------------------|
| `deepfake_deep_learning_model.pth` | Trained deep convolutional neural network model for deepfake detection using EfficientNet for image classification. |
| `deepfake_ml_model_final.joblib` | Final Random Forest model, trained using the full dataset with the best hyperparameters identified. |
| `deepfake_ml_model_optimized.joblib` | Intermediate Random Forest model obtained from hyperparameter tuning before full dataset training. |

### 📂 Scripts/
| File | Description |
|-------------------------|------------------------------------------------------------|
| `dataset.py` | Converts images to PyTorch tensors and prepares them for deep learning training. |
| `frame_extraction.py` | Script to extract frames from deepfake videos for training/testing purposes. |

### 📂 Data (External)
> **Note:** The dataset is too large to be stored in this repository. Instead, it is hosted on [Google Drive](https://drive.google.com/file/d/11oR8laHIBtOpCAId5JJv8KD4AF-5hPfS/view?usp=drive_link).

---

## **Video Presentation**
🎥 [Click here to watch the video presentation](https://drive.google.com/file/d/1bE7AVpmgT-eBt4grqQsNTz4gEhVOo7G1/view?usp=sharing)  

---

## **Dataset Information**
The dataset consists of **frames extracted from deepfake videos**. The full dataset is **too large to upload directly** but is available via [Google Drive](https://drive.google.com/file/d/11oR8laHIBtOpCAId5JJv8KD4AF-5hPfS/view?usp=drive_link).

📁 **[Here is the data source](https://www.kaggle.com/competitions/deepfake-detection-challenge)**  

> **Why This Dataset?**  
I chose the Kaggle Deepfake Detection dataset because it is one of the largest publicly available benchmark deepfake datasets. It contains a solid range of real and fake videos, thus making it suitable for training robust detection models. The dataset also includes pre-labeled metadata, removing the need for manual labeling. I used only the first three zipped data chunks due to computational constraints while making sure my models had a sufficiently large dataset for training.

---

### **Data Processing Steps**
**Extracted frames** from deepfake videos  
**Merged metadata** from `metadata.json` files  
**Labeled frames** as REAL or FAKE  
**Converted images into numerical feature vectors**  
**Standardized pixel values** for DL model. ML model used raw pixel values after tensor conversion and did not require standardization because random forests are not sensitive to feature scaling.

---

## **Machine Learning & Deep Learning Models**

### **Deep Learning Model (EfficientNet)**
- **Why EfficientNet?**  
  - Optimized architecture for image classification; very suitable for training a convolutional neural network on image data.
  - Faster training and high accuracy  

#### **Training Process**
1. Input frames  
2. Feature extraction using EfficientNet  
3. Train model on deepfake recognition  

#### **Evaluation Metrics**
- **Precision:** 0.9826  
- **Recall:** 0.9594  
- **F1-score:** 0.9708  

---

### **Machine Learning Model (Random Forest)**
- **Why Random Forest?**  
  - Handles high-dimensional image data efficiently  
  - Reduces overfitting by averaging multiple decision trees  
  - Manages class imbalance effectively  

#### **Evaluation Metrics**
- **Precision:** 0.9332  
- **Recall:** 0.9999  
- **F1-score:** 0.9654  

---

## **Key Findings**

### **Deep Learning Model (EfficientNet)**
**Strong overall accuracy and F1-score**  
**Balanced performance between precision and recall**  
**Slightly lower recall for real frames** → Some real frames misclassified as fake  

### **Machine Learning Model (Random Forest)**
**High recall for fake frames (detects deepfakes well)**  
**Major weakness:** High false positive rate (many real frames misclassified as fake)  
**ROC AUC (0.63) suggests only moderate separability between classes**  

---

## **Future Improvements**
**Improve real frame classification** by incorporating additional features, weights, and data

**Experiment with ensemble models** (e.g., combining DL and RF predictions)

**Fine-tune deep learning model further** to address class imbalance

---

## **Contact**
For any questions, feel free to reach out:  
📧 **[jonathanmonroe@uchicago.edu](mailto:jonathanmonroe@uchicago.edu)**  
