import os
import pyarrow.parquet as pq
import pandas as pd

# Folder containing prepared parquet files
folder_path = r"E:\1. Github\1. miRNA-RNA-Deep-Learning-Model\dataset\prepared_dataset"

# --- Auto-detect the latest parquet file ---
parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in {folder_path}")

# Sort by modified time (latest first)
parquet_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
file_path = os.path.join(folder_path, parquet_files[0])

print(f"Analyzing file: {file_path}")

# --- Load only the affinity column ---
df = pd.read_parquet(file_path, columns=["affinity"])

# --- Calculate stats ---
min_val = df["affinity"].min()
max_val = df["affinity"].max()
mean_val = df["affinity"].mean()
quartiles = df["affinity"].quantile([0.25, 0.5, 0.75])

# --- Print results ---
print("\n📊 Affinity Analysis")
print(f"Lowest value   : {min_val}")
print(f"Highest value  : {max_val}")
print(f"Average (mean) : {mean_val}")
print(f"25% quartile   : {quartiles.loc[0.25]}")
print(f"50% quartile   : {quartiles.loc[0.50]}  (Median)")
print(f"75% quartile   : {quartiles.loc[0.75]}")
