# create_safe_scaler.py
import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from safetensors.numpy import save_file

# --- CONFIGURATION ---
# The exact, full path to your input Parquet file.
INPUT_PARQUET_PATH = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\prepared_dataset\main\Prepared_Dataset_1756200284.parquet"

# The exact, full path where the final safe scaler file will be saved.
OUTPUT_SAFETENSORS_PATH = r"E:\backend_api\model_files\supreme_scaler.safetensors"
# --- END OF CONFIGURATION ---


def create_safe_scaler_file():
    """
    Fits a MinMaxScaler on a specific Parquet dataset and saves its parameters
    to a safe .safetensors file at a specific location.
    """
    print("--- Starting Safe Scaler Creation ---")

    # --- 1. Check Input and Prepare Output Directory ---
    if not os.path.exists(INPUT_PARQUET_PATH):
        print(f"FATAL ERROR: Input file not found at '{INPUT_PARQUET_PATH}'")
        return
        
    output_dir = os.path.dirname(OUTPUT_SAFETENSORS_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. Fit the Scaler ---
    print(f"\nFitting scaler using data from: {INPUT_PARQUET_PATH}")
    try:
        # These are the numerical features your model uses.
        numerical_features = ["gc_content", "dg", "conservation"]
        
        # Load only the necessary columns from the Parquet file to save memory
        df = pd.read_parquet(INPUT_PARQUET_PATH, columns=numerical_features)
        
        scaler = MinMaxScaler()
        scaler.fit(df)
        print("  - Scaler fitted successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not fit the scaler. {e}")
        return

    # --- 4. Extract Tensors and Save to a Safe .safetensors file ---
    # We save the underlying NumPy arrays (tensors) from the scaler object.
    scaler_tensors = {
        'min_': scaler.min_,
        'scale_': scaler.scale_
    }

    save_file(scaler_tensors, OUTPUT_SAFETENSORS_PATH)
        
    print(f"\n--- Safe Scaler Creation Complete ---")
    print(f"Scaler parameters have been saved to: {OUTPUT_SAFETENSORS_PATH}")


if __name__ == "__main__":
    create_safe_scaler_file()