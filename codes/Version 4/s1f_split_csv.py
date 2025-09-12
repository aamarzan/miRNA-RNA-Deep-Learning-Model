import json
import pandas as pd
from pathlib import Path
import math

# 1️⃣ Find config.json automatically
script_dir = Path(__file__).resolve().parent
config_path = None

for parent in [script_dir] + list(script_dir.parents):
    possible = parent / "config.json"
    if possible.exists():
        config_path = possible
        break

if not config_path:
    raise FileNotFoundError("config.json not found in script directory or any parent folders.")

# 2️⃣ Load config.json
with open(config_path, "r") as f:
    config = json.load(f)

print(f"✅ Loaded config from: {config_path}")

# 3️⃣ Get project_root from config
project_root = Path(config["project_root"])

# 4️⃣ Path to the for_convertion folder
conversion_folder = project_root / "dataset" / "prepared_dataset" / "for_convertion"

if not conversion_folder.exists():
    raise FileNotFoundError(f"Conversion folder not found: {conversion_folder}")

# 5️⃣ Find CSV files in the folder
csv_files = list(conversion_folder.glob("*.csv"))

if not csv_files:
    print("⚠️ No CSV files found in the conversion folder.")
else:
    for csv_file in csv_files:
        print(f"📂 Splitting file: {csv_file.name}")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Calculate chunk size
        total_rows = len(df)
        chunk_size = math.ceil(total_rows / 100)  # split into 100 parts
        
        # Split and save
        for i in range(100):
            start_row = i * chunk_size
            end_row = start_row + chunk_size
            chunk_df = df.iloc[start_row:end_row]
            
            if chunk_df.empty:
                break  # stop if no more rows
            
            part_filename = csv_file.stem + f"_part_{i+1}.csv"
            chunk_df.to_csv(conversion_folder / part_filename, index=False)
            print(f"✅ Saved: {part_filename}")
        
    print("🎯 Splitting complete!")
