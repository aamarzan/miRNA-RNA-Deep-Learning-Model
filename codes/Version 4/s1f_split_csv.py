import os
import pandas as pd
import math

# === CONFIG ===
folder_path = r"E:\1. Github\1. miRNA-RNA-Deep-Learning-Model\dataset\prepared_dataset"
num_parts = 10  # number of Excel files to create

# --- Find latest parquet file ---
parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in {folder_path}")

parquet_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
file_path = os.path.join(folder_path, parquet_files[0])

print(f"📂 Processing file: {file_path}")

# --- Load full dataset ---
df = pd.read_parquet(file_path)

# --- Save full dataset to Excel ---
full_excel_path = os.path.join(folder_path, "full_dataset.xlsx")
df.to_excel(full_excel_path, index=False)
print(f"✅ Saved full dataset to: {full_excel_path} ({len(df):,} rows)")

# --- Shuffle to ensure unbiased split ---
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Calculate chunk size ---
rows_per_part = math.ceil(len(df) / num_parts)

# --- Split and save ---
for i in range(num_parts):
    start_idx = i * rows_per_part
    end_idx = min((i + 1) * rows_per_part, len(df))
    part_df = df.iloc[start_idx:end_idx]
    
    out_path = os.path.join(folder_path, f"part_{i+1:02d}.xlsx")
    part_df.to_excel(out_path, index=False)
    print(f"✅ Saved {out_path} with {len(part_df):,} rows")

print("\n🎯 Done — full dataset and all parts saved as Excel files.")
