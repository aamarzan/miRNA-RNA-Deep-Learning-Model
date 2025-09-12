import pandas as pd
from pathlib import Path

# Set your directory path
base_dir = Path(r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\prepared_dataset\for_convertion")

# Loop through all .parquet files in the directory
for parquet_file in base_dir.glob("*.parquet"):
    try:
        # Create CSV file path with same name
        csv_file = parquet_file.with_suffix(".csv")
        
        # Read parquet and save as CSV
        df = pd.read_parquet(parquet_file)
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Converted: {parquet_file.name} → {csv_file.name}")
    except Exception as e:
        print(f"❌ Failed to convert {parquet_file.name}: {e}")

print("🎯 Conversion complete!")
