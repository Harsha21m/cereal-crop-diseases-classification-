# src/preprocess.py
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ---- Function to load all images ----
def load_data(dataset_dir="dataset", size=(128, 128)):
    X, y = [], []
    for crop in sorted(os.listdir(dataset_dir)):
        crop_path = os.path.join(dataset_dir, crop)
        if not os.path.isdir(crop_path):
            continue
        for disease in sorted(os.listdir(crop_path)):
            disease_path = os.path.join(crop_path, disease)
            if not os.path.isdir(disease_path):
                continue
            for file in os.listdir(disease_path):
                fpath = os.path.join(disease_path, file)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.resize(img, size)
                X.append(img)
                y.append(disease)
    X = np.array(X, dtype="float32") / 255.0  # normalize
    y = np.array(y)
    return X, y

# ---- Run directly to preprocess ----
if __name__ == "__main__":
    print("Loading images...")
    X, y = load_data("dataset")
    print(f"Loaded {len(X)} images.")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Save arrays & label encoder
    np.savez_compressed("models/preprocessed_data.npz",
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)
    joblib.dump(le, "models/label_encoder.joblib")

    print(" Data preprocessing completed and saved in /models/")
