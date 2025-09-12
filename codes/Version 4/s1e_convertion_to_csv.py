import json
import pandas as pd
from pathlib import Path

# 1️⃣ Start from script location and go upward until config.json is found
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

# 4️⃣ Build path to the for_convertion folder
conversion_folder = project_root / "dataset" / "prepared_dataset" / "for_convertion"

if not conversion_folder.exists():
    raise FileNotFoundError(f"Conversion folder not found: {conversion_folder}")

# 5️⃣ Convert all .parquet files in the folder
parquet_files = list(conversion_folder.glob("*.parquet"))

if not parquet_files:
    print("⚠️ No .parquet files found in the conversion folder.")
else:
    for parquet_file in parquet_files:
        try:
            csv_file = parquet_file.with_suffix(".csv")
            df = pd.read_parquet(parquet_file)
            df.to_csv(csv_file, index=False)
            print(f"✅ Converted: {parquet_file.name} → {csv_file.name}")
        except Exception as e:
            print(f"❌ Failed to convert {parquet_file.name}: {e}")

print("🎯 Conversion complete!")
