# src/hybrid_train.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# ---- Load Existing Model ----
print("🔹 Loading your existing model...")
old_model = load_model("models/crop_model.h5")
# old_model_input = old_model.input # Get the input of the old model

# ---- Load Preprocessed Data ----
print("🔹 Loading dataset...")
data = np.load("models/preprocessed_data.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ---- Load Pretrained MobileNetV2 ----
print("🔹 Loading pretrained MobileNetV2 backbone...")
hybrid_input = Input(shape=(128, 128, 3), name="hybrid_input")
old_model_input = Input(shape=(128, 128, 3), name="old_model_input") # New input for the old model branch
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_tensor=hybrid_input)
mobilenet_output = mobilenet(hybrid_input) # Apply mobilenet to the input
mobilenet.trainable = False  # freeze layers for now

# ---- Combine both models ----
old_model_current_output = old_model_input
for layer in old_model.layers[:-3]:
    old_model_current_output = layer(old_model_current_output)
x1 = old_model_current_output # features from your old model
x2 = mobilenet_output

merged = Concatenate()([GlobalAveragePooling2D()(x2), x1])
merged = Dense(256, activation='relu', name="hybrid_dense_1")(merged)
merged = Dropout(0.4)(merged)
final_output = Dense(num_classes, activation='softmax', name="hybrid_dense_output")(merged)

hybrid_model = Model(inputs=[hybrid_input, old_model_input], outputs=final_output)

# ---- Compile ----
hybrid_model.compile(optimizer=Adam(learning_rate=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# ---- Callbacks ----
checkpoint = ModelCheckpoint("models/hybrid_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ---- Train ----
print("Starting hybrid training...")
history = hybrid_model.fit(
    [X_train, X_train], y_train_cat,
    validation_data=([X_test, X_test], y_test_cat),
    epochs=15,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

print("model training completed and saved as models/hybrid_model.h5")
