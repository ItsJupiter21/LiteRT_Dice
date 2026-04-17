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


class DiceClassifier:
    def __init__(self, model_info: dict[str, Any]):
        """
        Loads the model into RAM once during initialization.
        """

        self.classes = model_info['classes']
        self.values = model_info['values']

        # 1. Load model and allocate RAM ONCE
        self.interpreter = litert.Interpreter(
            model_path=model_info['model_path'])
        self.interpreter.allocate_tensors()

        # 2. Cache the index pointers ONCE
        self.input_idx = self.interpreter.get_input_details()[0]['index']
        self.output_idx = self.interpreter.get_output_details()[0]['index']

        print(f"Loaded model as Tensorflow from {model_info['model_path']}")

    def classify(self, bgr_frame: np.ndarray) -> tuple[str, int, float]:
        """
        Runs extremely fast because the model is already waiting in RAM.
        """
        # Convert and resize
        if bgr_frame.shape != (128, 128, 3):
            bgr_frame = cv2.resize(bgr_frame, (128, 128))

        img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img.astype(np.float32), axis=0)

        # Run inference using cached indexes
        self.interpreter.set_tensor(self.input_idx, input_data)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_idx)[0]

        # Results
        p_idx = np.argmax(probs)
        return self.classes[p_idx], self.values[p_idx], probs[p_idx]

# --- Usage ---
# Initialize ONCE at the start of your script
# my_classifier = OptimizedLiteRTClassifier(dice_types["d6"])

# Call inside your loop (e.g., while True: reading from webcam)
# label, value, conf = my_classifier.classify(img)
