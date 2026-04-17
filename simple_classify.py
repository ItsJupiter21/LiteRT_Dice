import cv2

# only for the test
from models import DICE_TYPES
import pathlib
from time import time

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

try:
    from classifier_Tensorflow import DiceClassifier
except ImportError:
    print("failed to import using Tensorflow, using the LiteRT One.")
    from classifier import DiceClassifier

dice_type = 'd6_pips'
classifer = DiceClassifier(DICE_TYPES[dice_type])

base_dir = pathlib.Path(f"{dice_type}/")
base_dir = pathlib.Path(f"tests/d6/known")
''
subdirs = DICE_TYPES[dice_type]["classes"]
count = 0
starttime = time()
succ = 0
fail = 0
for sub in subdirs:
    # Join the base path with the subdirectory name
    current_path = base_dir / sub

    # Check if the path actually exists and is a directory to avoid errors
    if current_path.is_dir():
        print(f'Testing directory: {sub}')
        # Iterate over files within this specific subdirectory
        for file in current_path.iterdir():
            if file.is_file():
                count += 1
                proctime = time()
                img = cv2.imread(str(file))
                assert img is not None, f"Failed to load image: {file}"
                label, _, conf = classifer.classify(img)
                time_diff = time() - proctime
                if sub == label:
                    succ += 1
                    print(
                        f"---> ✅Predicted: {label:<5} with confidence {conf:.2f} for {file} , took {time_diff:.4f} seconds ")
                    if conf < 0.8:
                        print(
                            f"---> 🟡Predicted: {label:<5} with LOW confidence {conf:.2f} for {file} , took {time_diff:.4f} seconds ")
                else:
                    fail += 1
                    print(
                        f"--->❌ Predicted: {label:<5} with confidence {conf:.2f} for {file} , took {time_diff:.4f} seconds ")

    else:
        print(f"Warning: {current_path} does not exist.")

total_time = time() - starttime
print(f"failed {fail} times, sucseeded {succ} times")
print(
    f"Processed {count} images in {total_time:.2f} seconds, average {total_time/count:.4f} seconds per image")
