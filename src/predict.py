# src/predict.py
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model

# ---- Load model & label encoder ----
def load_assets():
    # model = load_model("models/crop_model.h5")
    model = load_model("models/hybrid_model.h5")
    le = joblib.load("models/label_encoder.joblib")
    return model, le

# ---- Preprocess uploaded image ----
def preprocess_image_bytes(file_bytes, size=(128, 128)):
    nparr = np.frombuffer(file_bytes, np.uint8)
    if nparr.size == 0:
        raise ValueError("The provided file bytes are empty or invalid, resulting in an empty numpy array.")
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("Bytes length:", len(file_bytes))
    if img is None:
        raise ValueError("Image decoding failed. Possibly invalid or corrupted image data.")
    img = cv2.resize(img, size)
    print("Resized image shape:", img.shape)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ---- Predict from uploaded file ----
def predict_from_bytes(file_bytes):
    model, le = load_assets()
    img = preprocess_image_bytes(file_bytes)
    preds = model.predict([img, img])[0]
    idx = int(np.argmax(preds))
    label = le.inverse_transform([idx])[0]
    confidence = float(preds[idx])

    # 🧩 Add threshold + Unknown handling
    if confidence < 0.9 or "Unknown" in label:
        label = "Unknown / Not a Leaf"
        confidence = confidence

    return label, confidence

