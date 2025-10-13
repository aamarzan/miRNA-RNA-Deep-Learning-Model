import os
import math
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


# === CONFIGURATION ===
# Path to your prepared_dataset folder
prepared_folder = r"E:\1. Github\1. miRNA-RNA-Deep-Learning-Model\dataset\prepared_dataset"

# Input Parquet file (auto-detect latest if empty)
input_filename = ""  # e.g., "training_combinations_part00000.parquet"
rows_per_part = 150_000   # target rows per output file
shuffle_buffer_size = 50_000  # how many rows to hold in memory before shuffling & writing
output_subfolder = "split_shuffled_parts"

# === SCRIPT START ===
print("--- Starting Memory-Safe Split with Shuffle ---")

# 1. Detect input file
if not input_filename:
    parquet_files = [f for f in os.listdir(prepared_folder) if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError("No Parquet files found in prepared_dataset folder.")
    input_filename = sorted(parquet_files)[-1]
    print(f"  - Auto-detected latest file: {input_filename}")

input_path = os.path.join(prepared_folder, input_filename)
output_folder = os.path.join(prepared_folder, output_subfolder)
os.makedirs(output_folder, exist_ok=True)

# 2. Open Parquet file
pf = pq.ParquetFile(input_path)
total_rows = pf.metadata.num_rows
print(f"  - Total rows: {total_rows:,}")

# 3. Prepare for streaming shuffle
buffer_df = pd.DataFrame()
part_counter = 0
rows_written_in_part = 0
current_part_rows = []

def flush_part(rows_list, part_id):
    """Write accumulated rows to a Parquet file."""
    if not rows_list:
        return
    df_out = pd.concat(rows_list, ignore_index=True)
    out_path = os.path.join(output_folder, f"{os.path.splitext(input_filename)[0]}_part{part_id:03d}.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"  - Saved {out_path} ({len(df_out):,} rows)")

# 4. Stream row groups in batches
for batch in pf.iter_batches(batch_size=shuffle_buffer_size):
    df_batch = batch.to_pandas()
    # Shuffle this batch
    df_batch = df_batch.sample(frac=1, random_state=np.random.randint(0, 1_000_000)).reset_index(drop=True)

    # Append to current part
    current_part_rows.append(df_batch)
    rows_written_in_part += len(df_batch)

    # If current part reached target size, flush it
    if rows_written_in_part >= rows_per_part:
        part_counter += 1
        flush_part(current_part_rows, part_counter)
        current_part_rows = []
        rows_written_in_part = 0

# 5. Flush any remaining rows
if current_part_rows:
    part_counter += 1
    flush_part(current_part_rows, part_counter)

print(f"\n✅ Done. Created {part_counter} shuffled parts in: {output_folder}")