import numpy as np
import tensorflow as tf
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


class DiceClassifier:
    def __init__(self, model_info: dict[str, Any]):
        """
        Loads the TensorFlow model into RAM once during initialization.
        """
        self.classes = model_info['classes']
        self.values = model_info['values']

        # 1. Load the full Keras/SavedModel model ONCE
        # 'model_path' should point to a SavedModel folder or .keras/.h5 file
        self.model = tf.keras.models.load_model(
            model_info['model_path_keras'])
        print(f"Loaded model as Tensorflow from {model_info['model_path']}")

    def classify(self, bgr_frame: np.ndarray) -> tuple[str, int, float]:
        """
        Runs fast because the model is already waiting in RAM.
        """
        # Convert and resize
        if bgr_frame.shape != (128, 128, 3):
            bgr_frame = cv2.resize(bgr_frame, (128, 128))

        img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img.astype(np.float32), axis=0)

        # Run inference
        # Note: Calling self.model() directly is faster than self.model.predict()
        # for single-image inference loops because it bypasses tf.data.Dataset overhead.
        predictions = self.model(input_data, training=False)
        probs = predictions[0].numpy()

        # Results
        p_idx = np.argmax(probs)
        return self.classes[p_idx], self.values[p_idx], float(probs[p_idx])

# --- Usage ---
# Initialize ONCE at the start of your script
# my_classifier = DiceClassifier(dice_types["d6"])

# Call inside your loop (e.g., while True: reading from webcam)
# label, value, conf = my_classifier.classify(img)
