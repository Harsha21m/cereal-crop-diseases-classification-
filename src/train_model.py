# src/train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# ---- Load preprocessed data ----
data = np.load("models/preprocessed_data.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

num_classes = len(np.unique(y_train))


y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ----  CNN Model ----
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(num_classes, activation="softmax")
])

# ---- Compile ----
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---- Callbacks ----
checkpoint = ModelCheckpoint("models/crop_model.h5", save_best_only=True,
                              monitor="val_accuracy", mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ---- Train ----
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=25,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

print("Training completed. Model saved as models/crop_model.h5")
