import numpy as np
import ai_edge_litert.interpreter as litert
import cv2

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False
if TYPE_CHECKING:
    from cv2.typing import MatLike
    from typing import Literal
    import numpy.typing as npt
else:
    MatLike = any

from typing import Any


def classify_dice(bgr_frame: MatLike, model: dict[str, Any]) -> tuple[str, int, float]:
    '''
    uses the provided model to classify the input image of a die and returns the predicted class name, dice value, and confidence score.

    Parameters:

        image: Input image as a NumPy array (BGR format as read by OpenCV), expected to be of shape (128, 128, 3).
        model: A dictionary containing the model information, including:
            - "classes": List of class names.
            - "values": List of corresponding dice values.
            - "model_path": Path to the TFLite model file.

    Returns:
        class_name: The predicted class name as a string.
        dice_value: The corresponding dice value as an integer.
        confidence: The confidence score of the prediction (float between 0 and 1).
    '''

    # 1. Load Model
    interpreter = litert.Interpreter(model['model_path'])
    interpreter.allocate_tensors()

    # Convert BGR to RGB
    if bgr_frame.shape != (128, 128, 3):

        print(
            f"Warning: Unexpected image shape {bgr_frame.shape}, expected (128, 128, 3). Attempting to resize.")
        bgr_frame = cv2.resize(bgr_frame, (128, 128))

    img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Normalize and add Batch dimension
    input_data = np.expand_dims(img.astype(np.float32), axis=0)

    # 3. Run Inference
    input_idx = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_idx, input_data)

    interpreter.invoke()

    # 4. Results
    output_idx = interpreter.get_output_details()[0]['index']
    probs = interpreter.get_tensor(output_idx)[0]

    p_idx = np.argmax(probs)

    class_name = model['classes'][p_idx]
    dice_value = model['values'][p_idx]

    confidence = probs[p_idx]

    return class_name, dice_value, confidence
