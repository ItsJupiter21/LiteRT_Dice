import tensorflow as tf
import pathlib
import os

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DICE_TYPE = "d6"
DATASET_DIR = pathlib.Path(DICE_TYPE)
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 500

print("Loading dataset...")

# ==========================================
# 2. Data Loading & Splitting (Updated)
# ==========================================
# Define any folders you want to ignore
EXCLUDE_DIRS = ["all-rolls"]

# Programmatically get all valid subdirectories in d6, ignoring the exceptions
valid_classes = [
    d.name for d in DATASET_DIR.iterdir()
    if d.is_dir() and d.name not in EXCLUDE_DIRS
]
valid_classes.sort()  # Sort alphabetically to ensure consistent class indices

print(f"Targeting specific classes: {valid_classes}")

# Pass the valid_classes list to the 'class_names' argument
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=valid_classes  # <--- This forces Keras to ignore 'all-rolls'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=valid_classes  # <--- Must match training
)

class_names = train_ds.class_names
print(f"Successfully loaded classes: {class_names}")


# ==========================================
# 3. Performance Optimization
# ==========================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 4. Model Architecture Construction
# ==========================================
# Data augmentation helps prevent overfitting on small datasets
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

# A lightweight CNN suitable for edge devices
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),  # Normalize pixel values to [0, 1]

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. Training the Model
# ==========================================
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================================
# 6. Conversion to LiteRT (TFLite)
# ==========================================
print("\nConverting model to LiteRT format...")

# Convert the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply default optimizations to quantize the model (reduces size and improves speed on edge)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the model to disk
tflite_filename = f"{DICE_TYPE}_classifier.tflite"
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"\nSuccess! LiteRT model saved as '{tflite_filename}'.")
print("You can now deploy this model to mobile or embedded edge devices.")
