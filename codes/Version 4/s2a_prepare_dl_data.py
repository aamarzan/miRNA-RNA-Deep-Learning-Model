# 2_deep_learning_data_preparation.py (UPGRADED WITH TIMERS AND PROGRESS BARS)
import os
import pandas as pd
import numpy as np
import json
import time
import joblib
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyarrow.parquet as pq
from tqdm import tqdm


# --- Configuration Loader ---
def load_config(config_path=None):
    """Loads and returns the configuration file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

# --- Helper Functions ---
def one_hot_encode_sequence(sequence, max_len, nucleotide_map):
    """One-hot encodes a single sequence."""
    encoded_seq = np.zeros((max_len, len(nucleotide_map)), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, nucleotide_map.get(char.upper(), len(nucleotide_map) - 1)] = 1 # Default to 'N'
    return encoded_seq

# --- Main Processing Function ---
def main():
    total_start_time = time.time()
    print("--- Starting Data Preparation for Deep Learning (Memory-Safe) ---")
    
    # --- Step 1: Load Config and Define Paths ---
    config = load_config()
    params = {**config['processing_parameters'], **config['training_parameters']}
    
    project_root = config['project_root']
    prepared_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['prepared_subfolder'])
    output_dl_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    os.makedirs(output_dl_folder, exist_ok=True)

    # --- Step 2: Auto-detect the most recent Parquet file ---
    print(f"\nScanning for datasets in: {prepared_folder}")
    try:
        prepared_files = [f for f in os.listdir(prepared_folder) if f.endswith('.parquet')]
        if not prepared_files: raise FileNotFoundError
        prepared_dataset_filename = sorted(prepared_files)[-1]
        prepared_dataset_path = os.path.join(prepared_folder, prepared_dataset_filename)
        print(f"  - Using latest dataset: {prepared_dataset_filename}")
    except FileNotFoundError:
        print(f"  - FATAL ERROR: No .parquet files found. Please run Stage 1 first.")
        exit()

    # --- Step 3: Fit Scaler in a Memory-Safe Way ---
    step1_start_time = time.time()
    print("\nStep 1: Fitting the scaler on numerical features...")
    numerical_features = params.get('numerical_features', ['gc_content', 'dg', 'conservation'])
    try:
        numerical_df = pd.read_parquet(prepared_dataset_path, columns=numerical_features)
        scaler = MinMaxScaler()
        scaler.fit(numerical_df)
        joblib.dump(scaler, os.path.join(output_dl_folder, 'minmax_scaler.pkl'))
        print(f"  - Scaler fitted on {len(numerical_df)} rows and saved.")
        del numerical_df
    except Exception as e:
        print(f"  - FATAL ERROR: Could not read numerical features to fit scaler: {e}")
        exit()
    step1_time = time.time() - step1_start_time
    print(f"  - Time for Step 1: {step1_time:.2f} seconds")

    # --- Step 4: Create Train-Test Split based on Indices ---
    step2_start_time = time.time()
    print("\nStep 2: Creating train-test split indices...")
    parquet_file = pq.ParquetFile(prepared_dataset_path)
    num_rows = parquet_file.metadata.num_rows
    indices = np.arange(num_rows)
    train_indices, test_indices = train_test_split(indices, test_size=params.get('test_split_ratio', 0.2), random_state=42)
    print(f"  - Total samples: {num_rows}")
    print(f"  - Training samples: {len(train_indices)}, Testing samples: {len(test_indices)}")
    
    sample_assignment = np.empty(num_rows, dtype='U5')
    sample_assignment[train_indices] = 'train'
    sample_assignment[test_indices] = 'test'
    del indices, train_indices, test_indices
    step2_time = time.time() - step2_start_time
    print(f"  - Time for Step 2: {step2_time:.2f} seconds")


    # --- Step 5: Process and Save Datasets in Batches ---
    step3_start_time = time.time()
    print("\nStep 3: Processing data in memory-safe batches...")
    
    pad_params = params['sequence_padding']
    max_primary_len, max_target_len, max_competitor_len = pad_params['max_primary_len'], pad_params['max_target_len'], pad_params['max_competitor_len']
    target_feature = params.get('target_feature', 'affinity')
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    
    def process_chunk(df_chunk, scaler_obj):
        y = df_chunk[target_feature].values.astype(np.float32)
        X = {
            'primary_sequence_input': np.array([one_hot_encode_sequence(seq, max_primary_len, nucleotide_map) for seq in df_chunk['primary_sequence']]),
            'target_sequence_input': np.array([one_hot_encode_sequence(seq, max_target_len, nucleotide_map) for seq in df_chunk['target_sequence']]),
            'competitor_sequence_input': np.array([one_hot_encode_sequence(seq, max_competitor_len, nucleotide_map) for seq in df_chunk['competitor_sequence']]),
            'primary_structure_input': np.expand_dims(pad_sequences(df_chunk['structure_vector'].apply(json.loads).tolist(), maxlen=max_primary_len, padding='post', dtype='float32'), axis=-1),
            'numerical_features_input': scaler_obj.transform(df_chunk[numerical_features])
        }
        return X, y

    dp_params = config.get('data_processing', {})
    batch_size = dp_params.get('batch_size_parquet', 1000)
    batch_iterator = parquet_file.iter_batches(batch_size=batch_size)
    total_batches = math.ceil(num_rows / batch_size)
    
    # Initialize lists to hold results
    first_batch_df = next(parquet_file.iter_batches(batch_size=1)).to_pandas()
    sample_X, _ = process_chunk(first_batch_df, scaler)
    X_train_batches, y_train_batches = {key: [] for key in sample_X}, []
    X_test_batches, y_test_batches = {key: [] for key in sample_X}, []
    
    processed_rows = 0
    # ⭐ NEW: Wrap the batch iterator with tqdm for a progress bar
    for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), total=total_batches, desc="  Processing batches"):
        df = batch.to_pandas()
        
        batch_indices = np.arange(processed_rows, processed_rows + len(df))
        assignments = sample_assignment[batch_indices]
        
        train_mask = (assignments == 'train')
        test_mask = (assignments == 'test')

        if np.any(train_mask):
            X_train, y_train = process_chunk(df[train_mask], scaler)
            for key in X_train_batches: X_train_batches[key].append(X_train[key])
            y_train_batches.append(y_train)

        if np.any(test_mask):
            X_test, y_test = process_chunk(df[test_mask], scaler)
            for key in X_test_batches: X_test_batches[key].append(X_test[key])
            y_test_batches.append(y_test)
            
        processed_rows += len(df)
    
    step3_time = time.time() - step3_start_time
    print(f"  - Time for Step 3: {step3_time:.2f} seconds")


    # --- Step 6: Finalize and Save Arrays ---
    step4_start_time = time.time()
    print("\nStep 4: Concatenating and saving final compressed .npz arrays...")

    # Save training data
    if y_train_batches:
        y_train_final = np.concatenate(y_train_batches)
        y_train_final_transformed = np.sqrt(y_train_final)
        np.savez_compressed(os.path.join(output_dl_folder, 'y_train.npz'), data=y_train_final_transformed)
        # ⭐ NEW: Wrap the saving loop with tqdm
        for key in tqdm(X_train_batches.keys(), desc="  Saving train arrays"):
            X_train_final = np.concatenate(X_train_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_train_{key}.npz'), data=X_train_final)
        print(f"  - Saved {len(y_train_final)} training samples.")

    # Save test data
    if y_test_batches:
        y_test_final = np.concatenate(y_test_batches)
        y_test_final_transformed = np.sqrt(y_test_final)
        np.savez_compressed(os.path.join(output_dl_folder, 'y_test.npz'), data=y_test_final_transformed)
        # ⭐ NEW: Wrap the saving loop with tqdm
        for key in tqdm(X_test_batches.keys(), desc="  Saving test arrays "):
            X_test_final = np.concatenate(X_test_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_test_{key}.npz'), data=X_test_final)
        print(f"  - Saved {len(y_test_final)} test samples.")
    
    step4_time = time.time() - step4_start_time
    print(f"  - Time for Step 4: {step4_time:.2f} seconds")

    total_time = time.time() - total_start_time
    print("\n--- Deep Learning Data Preparation Complete ---")
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()