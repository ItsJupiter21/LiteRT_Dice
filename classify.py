import tensorflow as tf
import numpy as np
import pathlib
import cv2
import os

# ==========================================
# 1. Configuration
# ==========================================
DICE_TYPE = "d6"
MODEL_PATH = f"{DICE_TYPE}_classifier.tflite"

TEST_DIR = pathlib.Path("d6/all-rolls")
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Note: These MUST be in alphabetical order to match training labels
CLASS_NAMES = ['five', 'four', 'one', 'six', 'three', 'two']

# ==========================================
# 2. Load the LiteRT (TFLite) Model
# ==========================================
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Run your training script first!")
    exit()

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def run_classifier():
    # Gather all images in the directory
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in extensions:
        image_files.extend(list(TEST_DIR.glob(ext)))

    if not image_files:
        print(f"No images found in: {TEST_DIR.absolute()}")
        return

    print(f"--- Starting Dice Review ---")
    print(f"Found {len(image_files)} images.")
    print("CONTROLS: Press any key for next image, 'q' to quit.")
    print("-" * 30)

    for img_path in image_files:
        # --- Preprocessing for Model ---
        try:
            # We use Keras load_img to ensure it matches training preprocessing
            img_keras = tf.keras.utils.load_img(
                img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img_keras)
            input_tensor = np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"Could not process {img_path.name}: {e}")
            continue

        # --- Inference ---
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Calculate results
        predicted_index = np.argmax(predictions)
        label = CLASS_NAMES[predicted_index]
        conf = predictions[predicted_index] * 100

        # --- Console Logging for Uncertainty ---
        # Almost all models are < 100% due to math, but we log it as requested
        if conf < 99:
            print(f"LOG: {img_path.name} | Pred: {label} | Conf: {conf:.2f}%")

        # --- OpenCV Visualization ---
        display_img = cv2.imread(str(img_path))
        if display_img is None:
            continue

        # Resize for better visibility on screen
        display_img = cv2.resize(
            display_img, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Update window and title
        window_name = "Dice Review"
        cv2.imshow(window_name, display_img)
        cv2.setWindowTitle(
            window_name, f"File: {img_path.name} | Prediction: {label} ({conf:.1f}%)")

        # Bring window to front
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # Wait for keypress - waitKey(0) pauses execution
        key = cv2.waitKey(0) & 0xFF

        # If 'q' or ESC (27) is pressed, exit
        if key == ord('q') or key == 27:
            print("Quitting...")
            break

    cv2.destroyAllWindows()
    print("Review finished.")


if __name__ == "__main__":
    run_classifier()
