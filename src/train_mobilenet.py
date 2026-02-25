# src/train_mobilenet.py

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------
# Load preprocessed data
# ---------------------------------------------
print("🔹 Loading dataset...")
data = np.load("models/preprocessed_data.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ---------------------------------------------
# Build MobileNetV2 Model
# ---------------------------------------------
print("🔹 Loading MobileNetV2 backbone...")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False   # Freeze layers initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------
# Callbacks
# ---------------------------------------------
checkpoint = ModelCheckpoint(
    "models/mobilenet_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ---------------------------------------------
# Train Phase 1 (Frozen Base Model)
# ---------------------------------------------
print("🔹 Training Phase 1 (feature extraction)...")
history1 = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=10,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# ---------------------------------------------
# Fine-Tuning Phase 2 (Unfreeze layers)
# ---------------------------------------------
print("🔹 Unfreezing base model layers...")
base_model.trainable = True

model.compile(
    optimizer=Adam(1e-5),  # small learning rate
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("🔹 Training Phase 2 (fine tuning)...")
history2 = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=5,
    batch_size=16,
    callbacks=[checkpoint, early_stop]
)

print("✔ Training completed — Saved as models/mobilenet_model.h5")
