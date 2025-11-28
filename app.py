import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import cv2

# --- Page Config ---
st.set_page_config(page_title="Kidney Disease Classifier", page_icon="🩺", layout="centered")

st.title("🩺 Kidney Disease Detection using YOLOv8")
st.write("Upload a kidney CT image to classify it as **Normal**, **Cyst**, **Tumor**, or **Stone**.")

# --- Load YOLOv8 model ---
model = YOLO("runs/detect/train/weights/best.pt")

# --- Reference Image Path ---
REFERENCE_IMAGE_PATH = "./Kidney-Disease-Detection-3/train/images/Cyst-5-_jpg.rf.3c5d40c9c57c4a19978bc3072424545b.jpg"

# ---------------- PREPROCESSING ----------------
def match_histogram(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)

def preprocess_image(test_path, ref_path):
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

    matched = match_histogram(test_img, ref_img).astype(np.uint8)

    matched = cv2.equalizeHist(matched)
    matched = cv2.GaussianBlur(matched, (3, 3), 0)
    matched_rgb = cv2.cvtColor(matched, cv2.COLOR_GRAY2BGR)

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_out.name, matched_rgb)

    return temp_out.name

# ---------------- STREAMLIT UI ----------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        image_path = temp_file.name

    st.info("🔧 Preprocessing image for better consistency...")
    preprocessed_path = preprocess_image(image_path, REFERENCE_IMAGE_PATH)

    with st.spinner("Classifying... ⏳"):
        results = model.predict(source=preprocessed_path, conf=0.1, imgsz=640, save=False)

    boxes = results[0].boxes
    names = results[0].names

    # If prediction exists
    if boxes is not None and len(boxes) > 0:
        annotated_img = results[0].plot()
        annotated_img = Image.fromarray(annotated_img[..., ::-1])

        cls_id = int(boxes.cls[0].item())
        confidence = float(boxes.conf[0].item()) * 100
        predicted_class = names[cls_id].capitalize()

        st.markdown(f"### 🧾 Prediction: **{predicted_class}** ({confidence:.1f}% confidence)")

        # ---------------- SIDE-BY-SIDE VIEW ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_path, caption="📤 Uploaded Test Image", use_column_width=True)

        with col2:
            st.image(annotated_img, caption="📸 YOLOv8 Prediction", use_column_width=True)

        # Show all detections
        with st.expander("Show all detections"):
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                conf = float(box.conf.item()) * 100
                st.write(f"**{i+1}. {names[cls_id].capitalize()} — {conf:.1f}%**")

    else:
        st.warning("⚠️ No disease detected in the image.")

    # Clean up
    os.remove(image_path)
    os.remove(preprocessed_path)
