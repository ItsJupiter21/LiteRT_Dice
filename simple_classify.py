import numpy as np
import ai_edge_litert.interpreter as litert
import cv2
import pathlib


def classify_cv2(bgr_frame, model_path):
    # 1. Load Model
    interpreter = litert.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 2. Preprocess OpenCV Image
    # Resize to match model input
    img = cv2.resize(bgr_frame, (128, 128))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize and add Batch dimension
    input_data = np.expand_dims(img.astype(np.float32), axis=0)

    # 3. Run Inference
    input_idx = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_idx, input_data)

    interpreter.invoke()

    # 4. Results
    output_idx = interpreter.get_output_details()[0]['index']
    probs = interpreter.get_tensor(output_idx)[0]

    classes = ['five', 'four', 'one', 'six', 'three', 'two']
    p_idx = np.argmax(probs)

    return classes[p_idx], probs[p_idx]


# Example usage:

files = pathlib.Path("./d6/all-rolls").glob("*.jpeg")
# randomise order
files = np.array(list(files))
np.random.shuffle(files)
for file in files:
    frame = cv2.imread(file)
    assert frame is not None, "Failed to load image. Check the path and file."

    frame = cv2.resize(frame, (128, 128))
    # cv2.imshow("Input Image", frame)
    # cv2.waitKey(0)
    label, conf = classify_cv2(frame, "d6_classifier.tflite")

    print(f"file {file} Predicted: {label} with confidence {conf:.2f}")
