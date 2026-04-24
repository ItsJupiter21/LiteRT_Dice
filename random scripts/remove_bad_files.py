import os
from pathlib import Path
from PIL import Image

# ⚠️ REPLACE THIS with the actual path to your training data directory
data_dir = Path(r"C:\Users\jupiter\Documents\LiteRT_Dice\train\d6")

removed_count = 0

for file_path in data_dir.rglob('*'):
    if file_path.is_file():
        try:
            # Attempt to open and verify the image
            img = Image.open(file_path)
            img.verify() # Check if it's broken
            
            # Check if the format is supported by TensorFlow
            if img.format not in ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP', 'JPEG2000']:
                raise ValueError(f"Unsupported format: {img.format}")
                
        except Exception as e:
            print(f"Removing invalid/corrupted file: {file_path} - Reason: {e}")
            os.remove(file_path)
            removed_count += 1

print(f"\nCleanup complete. Removed {removed_count} bad files.")