import pathlib
from classify import DiceClassifier
from models import DICE_TYPES
import cv2
from time import time

input_dir = pathlib.Path("tests/d6/unknown/")
output_dir = pathlib.Path("tests/d6/unknown/sorted/")
# Add 'unknown' for low confidence cases
subdirs = DICE_TYPES["d6"]["classes"] + ["unknown"]

for sub in subdirs:
    # create the dirs if they don't exist
    output_subdir = output_dir / sub
    output_subdir.mkdir(parents=True, exist_ok=True)

classifer = DiceClassifier(DICE_TYPES['d6'])


n = 0
starttime = time()
for file in input_dir.iterdir():

    if file.is_file():
        procstart = time()

        img = cv2.imread(str(file))
        assert img is not None, f"Failed to load image: {file}"
        label, _, conf = classifer.classify(img)
        proctime = time() - procstart
        n += 1
        if conf < 0.9:

           # print(f"Low confidence ({conf:.2f}) for {file}, skipping move.")
            print(
                f"LOW CONF: Predicted: {label} with confidence {conf:.2f} for {file}, took {proctime:.4f}")
            target_path = output_dir / "unknown" / file.name
        else:
            # Move the file to the corresponding subdirectory in the output directory
            target_path = output_dir / label / file.name
        file.rename(target_path)

totaltime = time() - starttime
print(f'{n} images in {totaltime:.2f} seconds, {n/totaltime:.4f} secs per image')
