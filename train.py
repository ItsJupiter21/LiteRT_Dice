import tensorflow as tf

from models import dice_types

from time import time
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DICE_TYPE = "d6"
if DICE_TYPE not in dice_types:
    print(
        f"Error: Unsupported DICE_TYPE '{DICE_TYPE}'. Available types: {list(dice_types.keys())}")
    exit()
DATASET_DIR = dice_types[DICE_TYPE]["dataset_dir"]
MODEL_PATH = dice_types[DICE_TYPE]["model_path"]

tflite_filename = f"{DICE_TYPE}_classifierv2.tflite"

FIG_PATH = f"training_metrics_{DICE_TYPE}_classifierv2.png"

BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 50

print("Loading dataset...")
starttime = time()
# ==========================================
# 2. Data Loading & Splitting (Updated)
# ==========================================

VALID_CLASSES = dice_types[DICE_TYPE]["classes"]

print(f"Targeting specific classes: {VALID_CLASSES}")

# Pass the valid_classes list to the 'class_names' argument
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=VALID_CLASSES  # <--- This forces Keras to ignore 'all-rolls'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=VALID_CLASSES  # <--- Must match training
)

class_names = train_ds.class_names
print(f"Successfully loaded classes: {class_names}")


# ==========================================
# 3. Performance Optimization
# ==========================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------
# STEP A: The CPU Data Pipeline (Color Shifting)
# ---------------------------------------------------------


def augment_colors(image, label):
    image = tf.image.random_hue(image, max_delta=0.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image, label


# Map it to training data only
train_ds = train_ds.map(augment_colors, num_parallel_calls=tf.data.AUTOTUNE)

# ---------------------------------------------------------
# STEP B: The GPU Model Pipeline (Geometry & Lighting)
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    # Geometric (Keep these!)
    tf.keras.layers.RandomRotation(1.0),
    tf.keras.layers.RandomZoom(0.1),

    # Lighting
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

# Build the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,          # <--- Geometric/Lighting happens here
    tf.keras.layers.Rescaling(1./255),

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
# visualisation
# ==========================================
for images, labels in train_ds.take(1):

    # 2. Pass the batch through your Keras augmentation block.
    # We must set training=True so the random layers know to activate.
    augmented_images = data_augmentation(images, training=True)

    # 3. Create a 3x3 grid to show 9 of the images
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        # Convert the tensor to a NumPy array and cast to uint8 (0-255)
        # Matplotlib requires this format to display RGB images correctly
        display_img = augmented_images[i].numpy().astype("uint8")

        plt.imshow(display_img)

        # Look up the actual string name (e.g., 'four') using the integer label
        plt.title(class_names[labels[i]])
        plt.axis("off")

    # Save it to your folder so you can inspect it
    plt.savefig("augmented_batch_preview.png")
    print("Success! Open 'augmented_batch_preview.png' to see your data.")

    # Break after the first batch so training can continue
    break

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

with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"\nSuccess! LiteRT model saved as '{tflite_filename}'.")
endtime = time()
time_diff = endtime - starttime
print(f"Total training and conversion time: {time_diff:.2f} seconds.")
print(f"Average time per epoch: {time_diff / EPOCHS:.2f} seconds.")


# Retrieve metrics from the history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIG_PATH)
print(f"Graph saved as {FIG_PATH}")
