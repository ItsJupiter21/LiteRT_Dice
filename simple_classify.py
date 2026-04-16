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
from classify import classify_dice


# Example usage:
if __name__ == "__main__":

    base_dir = pathlib.Path("tests/d6/known")
    subdirs = dice_types["d6"]["classes"]
    count = 0
    starttime = time()
    for sub in subdirs:
        # Join the base path with the subdirectory name
        current_path = base_dir / sub

        # Check if the path actually exists and is a directory to avoid errors
        if current_path.is_dir():
            print(f"\nProcessing: {current_path}")

            # Iterate over files within this specific subdirectory
            for file in current_path.iterdir():
                if file.is_file():
                    count += 1
                    img = cv2.imread(str(file))
                    assert img is not None, f"Failed to load image: {file}"
                    label, _, conf = classify_dice(img, dice_types["d6"])
                    time_diff = time() - starttime
                    print(
                        f"---> Predicted: {label:<5} with confidence {conf:.2f} for {file} , took {time_diff:.4f} seconds ")

        else:
            print(f"Warning: {current_path} does not exist.")

    total_time = time() - starttime
    print(
        f"Processed {count} images in {total_time:.2f} seconds, average {total_time/count:.4f} seconds per image")
