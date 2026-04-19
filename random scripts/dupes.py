import cv2
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
import os
from collections import defaultdict


def fast_interactive_cleanup(root_dir, hash_res=16):
    # 1. Setup
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = [p for p in Path(root_dir).rglob('*') if p.suffix.lower() in exts]

    # Using a dictionary to group EXACT hash matches (Threshold 0)
    hash_groups = defaultdict(list)

    print(
        f"--- Step 1: Hashing {len(paths)} images (Resolution: {hash_res}x{hash_res}) ---")
    for p in paths:
        try:
            with Image.open(p) as img:
                # Increasing hash_size makes it MUCH more sensitive to small details
                h = str(imagehash.phash(img, hash_size=hash_res))
                hash_groups[h].append(p)
        except Exception:
            continue

    # 2. Filter for only the actual duplicates
    duplicate_sets = [paths for paths in hash_groups.values()
                      if len(paths) > 1]

    if not duplicate_sets:
        print("No exact duplicates found with this hash resolution.")
        return

    print(
        f"--- Step 2: Auditing {len(duplicate_sets)} groups of duplicates ---")

    cv2.namedWindow("Duplicate Auditor", cv2.WINDOW_NORMAL)

    for group in duplicate_sets:
        # The first image in the group is our 'Original' (the one we keep)
        keep_path = group[0]

        # Every other image in the group is a candidate for deletion
        for delete_path in group[1:]:
            if not keep_path.exists() or not delete_path.exists():
                continue

            img_a = cv2.imread(str(keep_path))
            img_b = cv2.imread(str(delete_path))

            if img_a is None or img_b is None:
                continue

            # UI Construction
            img_a_res = cv2.resize(img_a, (640, 640))
            img_b_res = cv2.resize(img_b, (640, 640))
            canvas = np.hstack((img_a_res, img_b_res))

            # Text Overlays
            cv2.putText(canvas, f"KEEP: {keep_path.name}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(canvas, f"DELETE: {delete_path.name}", (650, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(canvas, "Press 'q' to CANCEL | Any other key to DELETE", (10, 620),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Duplicate Auditor", canvas)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print(f"Skipped: {delete_path.name}")
            else:
                print(f"Deleted: {delete_path.name}")
                os.remove(delete_path)

    cv2.destroyAllWindows()
    print("Audit finished.")


# --- Run ---
# hash_res=16 is the 'sweet spot' for dice.
# Go to 32 if you are still getting false positives.
fast_interactive_cleanup("./d6_pips", hash_res=16)
