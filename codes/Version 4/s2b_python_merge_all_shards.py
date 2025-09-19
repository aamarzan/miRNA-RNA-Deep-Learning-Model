import os
import re
import numpy as np
from collections import defaultdict

# Path to your processed_for_dl folder
dl_folder = r"E:\1. Github\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl"

# Create the 'final' subfolder for merged outputs
final_folder = os.path.join(dl_folder, "final")
os.makedirs(final_folder, exist_ok=True)

# Regex to capture: split, feature name, shard number
x_pattern = re.compile(r"^X_(train|test)_(.+)_shard\d+\.npz$")
y_pattern = re.compile(r"^y_(train|test)_shard\d+\.npz$")

# Step 1: Scan folder and group files
x_files = defaultdict(lambda: defaultdict(list))  # split -> feature -> [files]
y_files = defaultdict(list)  # split -> [files]

for fname in os.listdir(dl_folder):
    if m := x_pattern.match(fname):
        split, feature = m.groups()
        x_files[split][feature].append(fname)
    elif m := y_pattern.match(fname):
        split = m.group(1)
        y_files[split].append(fname)

# Step 2: Sort files for each feature
for split in x_files:
    for feature in x_files[split]:
        x_files[split][feature].sort()
for split in y_files:
    y_files[split].sort()

# Step 3: Merge and save into 'final' folder
for split in sorted(set(list(x_files.keys()) + list(y_files.keys()))):
    print(f"\n=== Merging split: {split.upper()} ===")

    # Merge X features
    merged_X = {}
    for feature, files in x_files[split].items():
        arrays = []
        for f in files:
            arr = np.load(os.path.join(dl_folder, f))['data']
            arrays.append(arr)
        merged_X[feature] = np.concatenate(arrays, axis=0)
        print(f"  Feature '{feature}': {len(files)} shards -> {merged_X[feature].shape}")

    # Merge y
    if split in y_files:
        y_arrays = []
        for f in y_files[split]:
            arr = np.load(os.path.join(dl_folder, f))['data']
            y_arrays.append(arr)
        merged_y = np.concatenate(y_arrays, axis=0)
        print(f"  Target y: {len(y_files[split])} shards -> {merged_y.shape}")
    else:
        merged_y = None
        print("  No y shards found for this split.")

    # Save merged files into final folder
    x_out_path = os.path.join(final_folder, f"X_{split}_merged.npz")
    y_out_path = os.path.join(final_folder, f"y_{split}_merged.npz")
    np.savez_compressed(x_out_path, **merged_X)
    if merged_y is not None:
        np.savez_compressed(y_out_path, data=merged_y)

    print(f"  ✅ Saved merged X to {x_out_path}")
    if merged_y is not None:
        print(f"  ✅ Saved merged y to {y_out_path}")

print(f"\n🎯 All splits merged successfully into: {final_folder}")
