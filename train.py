import tensorflow as tf
from models import DICE_TYPES
from time import time
import matplotlib.pyplot as plt
import math
import os

# ==========================================
# 1. Configuration & File Paths
# ==========================================
DICE_TYPE = "d6"
if DICE_TYPE not in DICE_TYPES:
    print(
        f"Error: Unsupported DICE_TYPE '{DICE_TYPE}'. Available types: {list(DICE_TYPES.keys())}")
    exit()

# Data Paths
DATASET_DIR = DICE_TYPES[DICE_TYPE]["dataset_dir"]
MODEL_PATH = DICE_TYPES[DICE_TYPE]["model_path"]

# Output File Paths
KERAS_FILENAME = f"{DICE_TYPE}_classifier.keras"
TFLITE_FILENAME = f"{DICE_TYPE}_classifier.tflite"

PREVIEW_FIG_PATH = f"images/augmented_batch_preview_{DICE_TYPE}.png"
METRICS_FIG_PATH = f"images/training_metrics_{DICE_TYPE}_classifier.png"

# Hyperparameters
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 40

print("Loading dataset...")
starttime = time()


# ==========================================
# 2. Data Loading & Splitting (The Mix)
# ==========================================
VALID_CLASSES = DICE_TYPES[DICE_TYPE]["classes"]
print(f"Targeting specific classes: {VALID_CLASSES}")

# A. Load the 24/7 Machine Data (80% Train, 20% Val)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=VALID_CLASSES
)

val_machine_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=VALID_CLASSES
)

class_names = train_ds.class_names
print(f"Successfully loaded classes: {class_names}")

# B. Load the Phone Data (100% Validation)
# You can add "phone_dir" to your DICE_TYPES dict, or it will default to a local folder
VAL_DIR = DICE_TYPES[DICE_TYPE].get("validation_dir", '')

if os.path.exists(VAL_DIR):
    print(f"\nMixing in real-world phone data from '{VAL_DIR}'...")
    val_phone_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_names=VALID_CLASSES,
        shuffle=False  # No need to shuffle validation data
    )

    # C. Concatenate (The Mix)
    val_ds = val_machine_ds.concatenate(val_phone_ds)
    print("Validation set now contains both machine and phone images.")
else:
    print(f"\nNo phone data directory found at '{VAL_DIR}'.")
    print("Using ONLY machine validation data.")
    val_ds = val_machine_ds

# ==========================================
# 3. CPU Data Pipeline (Color Shifting)
# ==========================================
AUTOTUNE = tf.data.AUTOTUNE

# Count the batches before we unbatch and lose the dataset length metadata
STEPS_PER_EPOCH = len(train_ds)


def augment_colors(image, label):
    # Randomly shifts the hue and saturation per-image to prevent color bias
    image = tf.image.random_hue(image, max_delta=0.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image, label


# Cache raw images FIRST, then shuffle
train_ds = train_ds.cache().shuffle(1000)

# Unbatch to individual images, apply unique colors, then re-batch
train_ds = train_ds.unbatch().map(
    augment_colors, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

# Repeat infinitely so Keras doesn't crash at the end of the epoch, then prefetch
train_ds = train_ds.repeat().prefetch(buffer_size=AUTOTUNE)

# Validation gets cached and prefetched ONLY (No color shifting)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 4. GPU Model Pipeline (Geometry & Light)
# ==========================================
data_augmentation = tf.keras.Sequential([
    # Geometric transformations with black background fill to prevent "kaleidoscope" corners
    tf.keras.layers.RandomRotation(1.0, fill_mode='constant', fill_value=0.0),
    tf.keras.layers.RandomZoom(0.1, fill_mode='constant', fill_value=0.0),

    # Lighting
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

# Build the model using Flatten for spatial awareness (required for reading fonts)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),

    # RESTORED: Flatten keeps pixel geometry intact so the model can read lines and loops
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
# 5. Visualisation (Sanity Check)
# ==========================================
print(f"\nGenerating augmented preview for full batch (Size: {BATCH_SIZE})...")

for images, labels in train_ds.take(1):
    augmented_images = data_augmentation(images, training=True)

    # 1. Dynamically calculate grid dimensions
    # We'll fix the number of columns to 8 and calculate needed rows
    cols = 8
    rows = math.ceil(len(images) / cols)

    # 2. Scale figsize based on grid size (approx 2 inches per image)
    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(len(images)):
        ax = plt.subplot(rows, cols, i + 1)

        # Convert to uint8 for plotting (Keras outputs floats [0-255] or [0-1])
        display_img = augmented_images[i].numpy().astype("uint8")

        plt.imshow(display_img)
        plt.title(class_names[labels[i]], fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(PREVIEW_FIG_PATH)
    print(f"Full batch preview saved to '{PREVIEW_FIG_PATH}'.")
    break
# ==========================================
# 6. Training the Model
# ==========================================
print("\nStarting training...")

# Early Stopping prevents overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,              # Drop this from 15 to 5 or 6
    min_delta=0.005,         # Ignore microscopic improvements
    restore_best_weights=True,
    verbose=1                # Prints exactly when and why it stopped
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[early_stop]
)

# ==========================================
# 7. Saving Both Model Formats
# ==========================================
print("\n--- Exporting Models ---")

# A. Save the Full Keras Model (.keras)
# Use this for high-precision desktop testing and retraining
print(f"1. Saving full Keras model: {KERAS_FILENAME}")
model.save(KERAS_FILENAME)

# B. Convert and Save LiteRT Model (.tflite)
# Use this for deployment on the Raspberry Pi
print(f"2. Converting to LiteRT: {TFLITE_FILENAME}")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Applies dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_FILENAME, 'wb') as f:
    f.write(tflite_model)

print(
    f"\nSuccess! Both models are now saved., {KERAS_FILENAME} and {TFLITE_FILENAME}")

endtime = time()
time_diff = endtime - starttime

actual_epochs = len(history.history['loss'])
print(f"Total training and conversion time: {time_diff:.2f} seconds.")
print(f"Average time per epoch: {time_diff / actual_epochs:.2f} seconds.")

# ==========================================
# 8. Metric Plotting
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(actual_epochs)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(METRICS_FIG_PATH)
print(f"Graph saved as {METRICS_FIG_PATH}")
