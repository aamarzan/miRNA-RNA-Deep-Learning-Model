# 2_deep_learning_data_preparation.py (Corrected Config-Driven, Memory-Safe Version)
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
    """
    Loads the configuration from a JSON file.
    If no path is given, it automatically finds 'config.json' in the same directory as the script.
    """
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
    start_time = time.time()
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
        if not prepared_files:
            raise FileNotFoundError
        prepared_dataset_filename = sorted(prepared_files)[-1]
        prepared_dataset_path = os.path.join(prepared_folder, prepared_dataset_filename)
        print(f"  - Using latest dataset: {prepared_dataset_filename}")
    except FileNotFoundError:
        print(f"  - FATAL ERROR: No .parquet files found. Please run Stage 1 first.")
        exit()

    # --- Step 3: Fit Scaler in a Memory-Safe Way ---
    print("\nStep 1: Fitting the scaler on numerical features...")
    # This pass only loads the numerical columns to minimize RAM usage
    numerical_features = params.get('numerical_features', ['gc_content', 'dg', 'conservation'])
    try:
        numerical_df = pd.read_parquet(prepared_dataset_path, columns=numerical_features)
        scaler = MinMaxScaler()
        scaler.fit(numerical_df)
        joblib.dump(scaler, os.path.join(output_dl_folder, 'minmax_scaler.pkl'))
        print(f"  - Scaler fitted on {len(numerical_df)} rows and saved.")
        del numerical_df # Free memory
    except Exception as e:
        print(f"  - FATAL ERROR: Could not read numerical features to fit scaler: {e}")
        exit()

    # --- Step 4: Create Train-Test Split based on Indices ---
    print("\nStep 2: Creating train-test split indices...")
    parquet_file = pq.ParquetFile(prepared_dataset_path)
    # <<< FIX: Access the number of rows from the metadata attribute
    num_rows = parquet_file.metadata.num_rows
    indices = np.arange(num_rows)
    train_indices, test_indices = train_test_split(indices, test_size=params.get('test_split_ratio', 0.2), random_state=42)
    print(f"  - Total samples: {num_rows}")
    print(f"  - Training samples: {len(train_indices)}, Testing samples: {len(test_indices)}")
    
    # Create a lookup array for efficient assignment
    sample_assignment = np.empty(num_rows, dtype='U5') # 'train' or 'test'
    sample_assignment[train_indices] = 'train'
    sample_assignment[test_indices] = 'test'
    del indices, train_indices, test_indices # Free memory

    # --- Step 5: Process and Save Datasets in Batches ---
    print("\nStep 3: Processing and saving datasets in memory-safe batches...")
    
    # Define constants from config
    pad_params = params['sequence_padding']
    max_primary_len = pad_params['max_primary_len']
    max_target_len = pad_params['max_target_len']
    max_competitor_len = pad_params['max_competitor_len']
    target_feature = params.get('target_feature', 'affinity')
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    
    # Helper to process a pandas DataFrame chunk
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

    # Use iter_batches for memory efficiency
    # Reads the batch size from the data_processing section of the config
    dp_params = config.get('data_processing', {})
    batch_iterator = parquet_file.iter_batches(batch_size=dp_params.get('batch_size_parquet', 1000))
    
    # Use lists to accumulate results before saving
    X_train_batches, y_train_batches = {key: [] for key in process_chunk(next(batch_iterator).to_pandas(), scaler)[0]}, []
    X_test_batches, y_test_batches = {key: [] for key in X_train_batches}, []
    
    # Reset iterator and process all batches
    batch_iterator = parquet_file.iter_batches(batch_size=params.get('batch_size', 10000))
    processed_rows = 0
    for batch in batch_iterator:
        df = batch.to_pandas()
        
        # Split dataframe based on pre-calculated indices
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
        print(f"  - Processed {processed_rows}/{num_rows} rows...", end='\r')

# --- Step 6: Finalize and Save Arrays ---
    print("\n\nStep 4: Concatenating and saving final compressed .npz arrays...")
    
    # Save training data
    if y_train_batches:
        y_train_final = np.concatenate(y_train_batches)
        # --- SAVE DIRECTLY TO NPZ ---
        np.savez_compressed(os.path.join(output_dl_folder, 'y_train.npz'), data=y_train_final)
        for key in X_train_batches:
            X_train_final = np.concatenate(X_train_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_train_{key}.npz'), data=X_train_final)
        print(f"  - Saved {len(y_train_final)} training samples.")

    # Save test data
    if y_test_batches:
        y_test_final = np.concatenate(y_test_batches)
        # --- SAVE DIRECTLY TO NPZ ---
        np.savez_compressed(os.path.join(output_dl_folder, 'y_test.npz'), data=y_test_final)
        for key in X_test_batches:
            X_test_final = np.concatenate(X_test_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_test_{key}.npz'), data=X_test_final)
        print(f"  - Saved {len(y_test_final)} test samples.")
        
    end_time = time.time()
    print("\n--- Deep Learning Data Preparation Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()