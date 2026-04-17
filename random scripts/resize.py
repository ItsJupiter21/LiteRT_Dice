import cv2
import os

dirs = ["tests/d6"]
size = 128

for folder in dirs:
    # Get all file names in the directory
    files = [f for f in os.listdir(
        folder) if os.path.isfile(os.path.join(folder, f))]

    for filename in files:
        img_path = os.path.join(folder, filename)

        # Load the image
        img = cv2.imread(img_path)

        if img is not None:
            # Resize to 128x128
            # interpolation=cv2.INTER_AREA is best for shrinking images
            resized_img = cv2.resize(
                img, (size, size), interpolation=cv2.INTER_AREA)

            # Save the resized image back to the same path
            cv2.imwrite(img_path, resized_img)
            print(f"Resized: {img_path}")
        else:
            print(f"Skipping non-image file: {filename}")

print("All images resized to 128x128.")
