import os

# Configuration
dirs = ['one', 'two', 'three', 'four', 'five', 'six', 'unknown']
prefix = 'z'


def rename_files_in_dirs(target_dirs, start_char):
    for directory in target_dirs:
        # Check if directory exists to avoid errors
        if not os.path.exists(directory):
            print(f"Directory '{directory}' not found. Skipping...")
            continue

        print(f"Processing directory: {directory}")

        # Iterate through files in the directory
        for filename in os.listdir(directory):
            old_path = os.path.join(directory, filename)

            # Ensure we are only renaming files, not sub-folders
            if os.path.isfile(old_path):
                new_name = start_char + filename
                new_path = os.path.join(directory, new_name)

                try:
                    os.rename(old_path, new_path)
                    print(f"  Renamed: {filename} -> {new_name}")
                except Exception as e:
                    print(f"  Error renaming {filename}: {e}")


if __name__ == "__main__":
    rename_files_in_dirs(dirs, prefix)
    print("\nTask complete!")
