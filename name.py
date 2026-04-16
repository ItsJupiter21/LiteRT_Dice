import os


# DO NOT USE HERE, USE BEFORE MOVING DATA TO TRAIN FOLDER

# Input Data
dirs = ['one', 'two', 'three', 'four', 'five', 'six']
numbers = [1, 2, 3, 4, 5, 6]
die_size = "d6"
die_type = "yellow"

for index, dir_name in enumerate(dirs):
    if not os.path.exists(dir_name):
        print(f"Directory '{dir_name}' not found. Skipping...")
        continue

    # Get the corresponding number for this directory
    num_val = numbers[index]

    # We sort them to ensure the 0000, 0001 sequence is predictable
    files = sorted([f for f in os.listdir(dir_name)
                   if os.path.isfile(os.path.join(dir_name, f))])

    print(f"Processing '{dir_name}' (Value: {num_val})...")

    for count, filename in enumerate(files):
        # Split extension to keep it (e.g., .jpeg)
        file_ext = os.path.splitext(filename)[1]

        new_name = f"{die_type}_{die_size}_{num_val}_{count:04d}{file_ext}"

        old_path = os.path.join(dir_name, filename)
        new_path = os.path.join(dir_name, new_name)

        os.rename(old_path, new_path)

print("done!!")
