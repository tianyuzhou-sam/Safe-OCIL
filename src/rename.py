import os
import glob

# Folder containing .mat files (use '.' for current directory)
folder = "examples/cartpole/data/results/"

# Find all .mat files that start with "results_"
mat_files = sorted(glob.glob(os.path.join(folder, "results_*.mat")))

# Rename sequentially
for i, old_file in enumerate(mat_files, start=1):
    new_file = os.path.join(folder, f"results_{i}.mat")
    print(f"Renaming {old_file} -> {new_file}")
    os.rename(old_file, new_file)
