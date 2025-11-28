# 🩺 Kidney Disease Prediction Using YOLOv8

A deep-learning based object detection system for identifying kidney
abnormalities such as stones, tumors, cysts, and normal kidney regions
from ultrasound images.

This project uses Ultralytics YOLOv8 for training, validation, and
inference. The repository includes training scripts, dataset structure,
model weights, and evaluation results.

## 📁 Project Structure

    Kidney-disease-prediction/
    │
    ├── data/
    │   ├── train/
    │   ├── valid/
    │   └── data.yaml
    │
    ├── models/
    │   └── yolov8n.pt (or custom weights)
    │
    ├── runs/
    │   └── detect/
    │       └── train/
    │
    ├── scripts/
    │   └── train.py
    │   └── inference.py
    │
    └── README.md

## 🚀 Features

-   Detection of four kidney classes:
    -   NORMAL
    -   STONE
    -   TUMOR
    -   CYSTS
-   Trained with YOLOv8 (Ultralytics)
-   High validation performance with detailed plots
-   Automatically generates bounding box predictions
-   Includes training logs, metrics, and confusion matrices

## 📊 Validation Metrics & Results

### **F1--Confidence Curve**

Shows optimal confidence threshold across classes.\
Best overall F1 = **0.88 at 0.257 confidence**.

### **Precision--Confidence Curve**

Overall precision reaches **1.00 at 0.629 confidence**.

### **Recall--Confidence Curve**

Recall remains strong, peaking at **0.98**.

### **Precision--Recall Curve (mAP@0.5)**

Class-wise mAP@0.5: - NORMAL: **0.995** - STONE: **0.850** - TUMOR:
**0.784** - CYSTS: **0.983**

Overall **mAP@0.5 = 0.903**

### **Confusion Matrix**

Shows prediction accuracy per class (raw and normalized).

## 🧠 Model Training

### 1️⃣ Install Dependencies

    pip install ultralytics
    pip install matplotlib numpy

### 2️⃣ Train the Model

    yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640

### 3️⃣ Run Inference

    yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images

Outputs will be saved to:

    runs/detect/predict/

## 📦 Dataset

Your dataset follows the YOLO format:

    data/
    ├── images/
    │   ├── train/
    │   └── valid/
    └── labels/
        ├── train/
        └── valid/

Modify **data.yaml** accordingly:

    train: data/images/train
    val: data/images/valid

    nc: 4
    names: ["NORMAL", "STONE", "TUMOR", "cysts"]

## 📈 Performance Summary

  Metric           Value
  ---------------- --------------
  mAP@0.5          0.903
  Best F1          0.88 @ 0.257
  Best Precision   1.00 @ 0.629
  Best Recall      0.98 @ 0.0

## 🛠 Tools & Technologies

-   Ultralytics YOLOv8\
-   Python\
-   Numpy, Matplotlib\
-   Google Colab / GPU acceleration\
-   LabelImg / Roboflow for dataset preparation

## 🌟 Applications

-   Early kidney disease diagnosis\
-   Automated ultrasound report support\
-   Real-time screening systems for hospitals

## 👤 Author

**Sai Sruthi Kancharla**\
GitHub: https://github.com/kancharlasaisruthi
