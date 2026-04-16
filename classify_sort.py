import pathlib
from classify import classify_dice
from models import dice_types
import cv2

input_dir = pathlib.Path("tests/d6/known2")
output_dir = pathlib.Path("tests/d6/known2/sorted")
# Add 'unknown' for low confidence cases
subdirs = dice_types["d6"]["classes"] + ["unknown"]

for sub in subdirs:
    # create the dirs if they don't exist
    output_subdir = output_dir / sub
    output_subdir.mkdir(parents=True, exist_ok=True)

for file in input_dir.iterdir():
    if file.is_file():
        img = cv2.imread(str(file))
        assert img is not None, f"Failed to load image: {file}"
        label, _, conf = classify_dice(img, dice_types["d6"])

        if conf < 0.6:

           # print(f"Low confidence ({conf:.2f}) for {file}, skipping move.")
            print(
                f"LOW CONF: Predicted: {label} with confidence {conf:.2f} for {file}")
            target_path = output_dir / "unknown" / file.name
        else:
            # Move the file to the corresponding subdirectory in the output directory
            target_path = output_dir / label / file.name
        file.rename(target_path)
