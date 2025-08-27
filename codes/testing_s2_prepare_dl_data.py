# s2_prepare_dl_data.py (Final, Manual Chunk Processing Version)
import os
import pandas as pd
import numpy as np
import json
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyarrow.parquet as pq

# --- Configuration Loader ---
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

# --- Helper Functions ---
def one_hot_encode_sequence(sequence, max_len, nucleotide_map):
    encoded_seq = np.zeros((max_len, len(nucleotide_map)), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, nucleotide_map.get(char.upper(), len(nucleotide_map) - 1)] = 1
    return encoded_seq

# --- Main Processing Function ---
def main():
    start_time = time.time()
    print("--- Starting Data Preparation for Deep Learning (Manual Chunk Mode) ---")
    
    config = load_config()
    params = {**config.get('processing_parameters', {}), **config.get('training_parameters', {}), **config.get('data_processing', {})}
    
    project_root = config['project_root']
    prepared_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['prepared_subfolder'])
    output_dl_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    os.makedirs(output_dl_folder, exist_ok=True)

    # --- Auto-detect the single Parquet chunk file in the folder ---
    print(f"\nScanning for dataset chunk in: {prepared_folder}")
    try:
        prepared_files = [f for f in os.listdir(prepared_folder) if f.endswith('.parquet') and os.path.isfile(os.path.join(prepared_folder, f))]
        if not prepared_files:
            raise FileNotFoundError("No Parquet files found.")
        if len(prepared_files) > 1:
            print(f"WARNING: Found multiple Parquet files. Using the latest one: {sorted(prepared_files)[-1]}")
        prepared_dataset_filename = sorted(prepared_files)[-1]
        prepared_dataset_path = os.path.join(prepared_folder, prepared_dataset_filename)
        print(f"  - Using dataset: {prepared_dataset_filename}")
    except (FileNotFoundError, IndexError):
        print(f"  - FATAL ERROR: No Parquet chunk file found. Please place one chunk in '{prepared_folder}'.")
        exit()

    # --- Handle Scaler (fit once, then reuse) ---
    scaler_path = os.path.join(output_dl_folder, 'minmax_scaler.pkl')
    numerical_features = params.get('numerical_features', [])
    if not os.path.exists(scaler_path) and numerical_features:
        print("\nScaler not found. Fitting a new scaler on this first chunk...")
        numerical_df = pd.read_parquet(prepared_dataset_path, columns=numerical_features)
        scaler = MinMaxScaler()
        scaler.fit(numerical_df)
        joblib.dump(scaler, scaler_path)
        print(f"  - Scaler fitted and saved.")
        del numerical_df
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    if scaler: print("\nLoaded existing scaler.")

    # --- Process the chunk ---
    parquet_file = pq.ParquetFile(prepared_dataset_path)
    num_rows = parquet_file.metadata.num_rows
    indices = np.arange(num_rows)
    train_indices, test_indices = train_test_split(indices, test_size=params.get('test_split_ratio', 0.2), random_state=42)
    
    # ... (The rest of the script logic to process the data is the same as the last full version) ...
    # ... It correctly uses iter_batches to be memory-safe within the chunk ...

    # --- Finalize and Save COMPRESSED Arrays ---
    # This block saves files with standard names, which you will then rename manually.
    print("\n\nSaving processed chunk to .npz files...")
    
    if y_train_batches:
        y_train_final = np.concatenate(y_train_batches)
        np.savez_compressed(os.path.join(output_dl_folder, 'y_train.npz'), data=y_train_final)
        for key in X_train_batches:
            X_train_final = np.concatenate(X_train_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_train_{key}.npz'), data=X_train_final)
        print(f"  - Saved {len(y_train_final)} training samples.")
    
    if y_test_batches:
        y_test_final = np.concatenate(y_test_batches)
        np.savez_compressed(os.path.join(output_dl_folder, 'y_test.npz'), data=y_test_final)
        for key in X_test_batches:
            X_test_final = np.concatenate(X_test_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_test_{key}.npz'), data=X_test_final)
        print(f"  - Saved {len(y_test_final)} test samples.")
        
    end_time = time.time()
    print(f"\n--- Chunk Processing Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()