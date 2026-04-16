import cv2

# only for the test
from models import dice_types
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
from classify import DiceClassifier


# Example usage:
if __name__ == "__main__":
    classifer = DiceClassifier(dice_types['d6'])

    base_dir = pathlib.Path("tests/d6/known")
    subdirs = dice_types["d6"]["classes"]
    count = 0
    starttime = time()
    succ = 0
    fail = 0
    for sub in subdirs:
        # Join the base path with the subdirectory name
        current_path = base_dir / sub

        # Check if the path actually exists and is a directory to avoid errors
        if current_path.is_dir():
            print(f"\nProcessing: {current_path}")
            print(sub)
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
                    else:
                        fail += 1
                        print(
                            f"--->❌ Predicted: {label:<5} with confidence {conf:.2f} for {file} , took {time_diff:.4f} seconds ")

        else:
            print(f"Warning: {current_path} does not exist.")

    total_time = time() - starttime
    print(f"failed {fail} times, sucseeded {succ} times")
    print(
        f"Processed {count} images in {total_time:.2f} seconds, average {count/total_time:.4f} seconds per image")
