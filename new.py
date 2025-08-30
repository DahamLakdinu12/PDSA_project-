# strawberry_cnn.py
# -------------------------------
# Train CNN Model for Strawberry Quality (Pick / Unpick)
# -------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# -------------------------------
# Dataset Configuration
# -------------------------------
dataset_path = "/Users/dahamlakdinu/Desktop/strawberryDataset"  # Change this path
image_size = (150, 150)
batch_size = 32

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

# Data generators with augmentation (helps accuracy)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# -------------------------------
# Build CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Train Model with EarlyStopping & Best Save
# -------------------------------
checkpoint = ModelCheckpoint("best_strawberry_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    epochs=15,
    validation_data=val_data,
    callbacks=[checkpoint, early_stop]
)

# -------------------------------
# Plot Accuracy & Loss
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# -------------------------------
# Webcam Capture & Prediction
# -------------------------------
def capture_and_predict(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("Press 'c' to capture an image, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("Webcam - Press c to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img = cv2.resize(frame, image_size)
            img = np.expand_dims(img, axis=0) / 255.0
            prediction = model.predict(img)

            prob = prediction[0][0]
            if prob > 0.6:
                print(f"✅ Good (Pick) | Confidence: {prob:.2f}")
            elif prob < 0.4:
                print(f"❌ Not Good (Unpick) | Confidence: {prob:.2f}")
            else:
                print(f"⚠️ Borderline Strawberry | Confidence: {prob:.2f} (Re-check needed)")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Load Best Model & Run Webcam
# -------------------------------
model = tf.keras.models.load_model("best_strawberry_model.h5")
capture_and_predict(model)

# -------------------------------
# Export to TensorFlow Lite (For Android App)
# -------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("strawberry_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model exported as strawberry_model.tflite (ready for Android)")
