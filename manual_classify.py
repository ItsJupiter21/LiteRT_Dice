#! /usr/bin/env python3

#  Original code Copyright (c) 2019 Andrew Lauritzen
# Licensed under the MIT License.
#  Modifications Copyright (c) 2026 Alex U
# Licensed under the MIT License.
#
import os
import sys
import cv2
from pathlib import Path
import shutil

# Settings
INPUT_DIR = 'C:/Users/jupiter/Desktop/tosort'
OUTPUT_DIR = 'C:/Users/jupiter/Desktop/sorted'
INPUT_EXT = '.jpeg'


###################################################################################################

KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_RIGHT = 2555904
KEY_LEFT = 2424832

###################################################################################################


cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

capture_list = list(
    Path(os.path.join(INPUT_DIR)).glob('*{}'.format(INPUT_EXT)))
print("Found {} files".format(len(capture_list)))
if not capture_list:
    print("No input files found in {}".format(INPUT_DIR))
    sys.exit(1)

capture_index = 0
last_capture_index = -1
capture_image = None
file_name = capture_list[capture_index]
base_file_name = os.path.basename(file_name)

while cv2.getWindowProperty('main1', 0) >= 0:
    if capture_index < 0:
        capture_index = 0
    if capture_index >= len(capture_list):
        print("No more captures available.")
        break

    if capture_index != last_capture_index:
        file_name = capture_list[capture_index]
        base_file_name = os.path.basename(file_name)

        if file_name.exists():
            capture_image = cv2.imread(str(file_name))
            if capture_image is None:
                print("Failed to load capture {}".format(base_file_name))
            else:
                capture_image = cv2.resize(capture_image, (400, 400))
                print("Loaded capture index {}".format(base_file_name))
        else:
            print("Capture {} not found!".format(base_file_name))

        last_capture_index = capture_index

    if capture_image is None:
        cv2.waitKeyEx(100)
        continue

    cv2.imshow('main1', capture_image)

    category = None

    key = cv2.waitKeyEx(10)
    if (key >= 0):
        if key == KEY_RIGHT:
            capture_index += 1
        elif key == KEY_LEFT:
            if (capture_index > 0):
                capture_index -= 1
        elif key == ord('1'):
            category = "one"
        elif key == ord('2'):
            category = "two"
        elif key == ord('3'):
            category = "three"
        elif key == ord('4'):
            category = "four"
        elif key == ord('5'):
            category = "five"
        elif key == ord('6'):
            category = "six"
        elif key == ord('7'):
            category = "seven"
        elif key == ord('8'):
            category = "eight"

        if category is not None:
            path = os.path.join(OUTPUT_DIR, category)
            if not os.path.exists(path):
                os.makedirs(path)

            output_file = os.path.join(path, base_file_name)
            if Path(output_file).exists():
                print("Cannot categorize {} as {}: output file {} exists!".format(
                    base_file_name, category, output_file))
            else:
                shutil.move(file_name, output_file)
                print("Categorized {} as {} ({})".format(
                    base_file_name, category, output_file))
                del capture_list[capture_index]
                capture_image = None
                last_capture_index = -1
                continue

cv2.destroyAllWindows()
