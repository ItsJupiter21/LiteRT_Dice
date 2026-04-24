#! /usr/bin/env python3

import os
import sys
import glob
import cv2

# Windows waitKeyEx arrow key codes
KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_RIGHT = 2555904
KEY_LEFT = 2424832

# Find the target CSV file
csv_files = glob.glob("*.csv")
if not csv_files:
    print("Error: No CSV file found in this directory.")
    sys.exit(1)

csv_filename = csv_files[0]
output_file = f"CORRECTIONS_{csv_filename}"

# Get all jpegs in current directory
images = glob.glob("*.jpeg") + glob.glob("*.jpg")

if not images:
    print("No images found in this directory.")
    sys.exit(1)

print(f"Found {len(images)} images to label.")
print("Press 1-6 to log the correct ROLL_VALUE. The image will auto-advance.")
print("DONT ENTER THE VALUE YOU SEE, YOU HAVE TO DO THE OPPOSITE FACE")
print("Press Left/Right arrows to navigate.")
print("Press 'q' or ESC to quit.")

# Open CSV for appending
with open(output_file, 'a') as f:
    f.write('FILE, ROLL_CORRECTED\n')

    cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

    capture_index = 0
    last_capture_index = -1
    capture_image = None

    while cv2.getWindowProperty('main1', 0) >= 0:
        # Keep index within bounds
        if capture_index < 0:
            capture_index = 0
        if capture_index >= len(images):
            print("End of image list reached.")
            capture_index = len(images) - 1

        # Load new image if the index changed
        if capture_index != last_capture_index:
            img_path = images[capture_index]
            base_file_name = os.path.basename(img_path)

            capture_image = cv2.imread(img_path)
            if capture_image is None:
                print(f"Failed to load capture {base_file_name}")
            else:
                # Resize for easier viewing
                capture_image = cv2.resize(capture_image, (600, 600))

                # Add text overlay to see the filename and progress
                cv2.putText(capture_image, f"{base_file_name} ({capture_index + 1}/{len(images)})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            last_capture_index = capture_index

        if capture_image is None:
            cv2.waitKeyEx(100)
            continue

        cv2.imshow('main1', capture_image)

        # Listen for keystrokes
        key = cv2.waitKeyEx(10)

        if key >= 0:
            if key == KEY_RIGHT:
                capture_index += 1
            elif key == KEY_LEFT:
                if capture_index > 0:
                    capture_index -= 1
            elif key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif ord('1') <= key <= ord('6'):
                # Extract the number pressed
                val = chr(key)

                # Write to CSV and flush immediately
                f.write(f"{base_file_name},{val}\n")
                f.flush()
                print(f"Logged {base_file_name} -> {val}")

                # Auto-advance to the next image
                capture_index += 1

cv2.destroyAllWindows()
print(f"\nDone! Corrections saved to: {output_file}")
