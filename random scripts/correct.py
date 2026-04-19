import os
import glob

csv_files = glob.glob("*.csv")
if not csv_files:
    print("Error: No CSV file found in this directory.")
    exit()

# Use the first CSV found
csv_filename = csv_files[0]
# Configuration
output_file = f"CORRECTIONS_{csv_filename}"

# Get all jpegs in current directory
images = glob.glob("*.jpeg") + glob.glob("*.jpg")

print(f"Found {len(images)} images to label.")
print("Enter the correct ROLL_VALUE (1-6). Press Enter to skip, 'q' to quit.")
print("DONT ENTER THE VALUE YOU SEE, YOU HAVE TO DO THE OPPOSITE FACE")

with open(output_file, 'a') as f:
        f.write('FILE, ROLL_CORRECTED')

    for img_path in images:
        fname = os.path.basename(img_path)
        # Simple prompt
        val = input(f"File [{fname}] -> Correct Value: ").strip()
        
        if val.lower() == 'q':
            break
        elif val:
            f.write(f"{fname},{val}\n")
            f.flush()

print(f"\nDone! Corrections saved to: {output_file}")