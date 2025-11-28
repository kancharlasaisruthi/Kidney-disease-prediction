# 🩺 Kidney Disease Prediction Using YOLOv8

A deep-learning based object detection system for identifying kidney
abnormalities such as stones, tumors, cysts, and normal kidney regions
from ultrasound images.

This project uses Ultralytics YOLOv8 for training, validation, and
inference. The repository includes training scripts, dataset structure,
model weights, and evaluation results.



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

## Working Demo using Streamlit
- allows to upload CT-Scan image from system
<img width="803" height="422" alt="image" src="https://github.com/user-attachments/assets/8f2ef53e-5068-4d6a-93df-18cbfbed0edd" />
<img width="564" height="576" alt="image" src="https://github.com/user-attachments/assets/a1d60d4e-8453-4d1c-9e9c-e566b408e01a" />
<img width="589" height="536" alt="image" src="https://github.com/user-attachments/assets/3b85ba20-5e49-4378-a57b-06877dec0d11" />
<img width="601" height="562" alt="image" src="https://github.com/user-attachments/assets/9835c0c7-c5ea-4261-9a78-a1e862d9a011" />
<img width="608" height="530" alt="image" src="https://github.com/user-attachments/assets/dc465a78-8df3-458c-b0ad-c1b2c9db4cea" />

## 📊 Validation Metrics & Results

### **F1--Confidence Curve**

Shows optimal confidence threshold across classes.\
Best overall F1 = **0.88 at 0.257 confidence**.

<img width="712" height="466" alt="image" src="https://github.com/user-attachments/assets/bd2dcd62-2c2b-43df-ad58-23ac86d2af04" />

### **Precision--Confidence Curve**

Overall precision reaches **1.00 at 0.629 confidence**.

<img width="694" height="452" alt="image" src="https://github.com/user-attachments/assets/176b5c98-b925-4772-a415-a68229d62c65" />

### **Recall--Confidence Curve**

Recall remains strong, peaking at **0.98**.

<img width="681" height="460" alt="image" src="https://github.com/user-attachments/assets/93933a63-3761-4ac4-b386-2db956947646" />

### **Precision--Recall Curve (mAP@0.5)**

Class-wise mAP@0.5: - NORMAL: **0.995** - STONE: **0.850** - TUMOR:
**0.784** - CYSTS: **0.983**

Overall **mAP@0.5 = 0.903**

<img width="685" height="453" alt="image" src="https://github.com/user-attachments/assets/61e01f29-37e8-447b-bd9f-bb68dcb625a2" />

### **Confusion Matrix**

Shows prediction accuracy per class (raw and normalized).
<img width="545" height="445" alt="image" src="https://github.com/user-attachments/assets/00fc1449-1208-4728-8009-ea7ac4842978" />

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

dataset follows the YOLO format:

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
