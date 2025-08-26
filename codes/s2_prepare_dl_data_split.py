# s2_prepare_dl_data.py (Final, Chunk-Processing Version)
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
import argparse

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
def main(args):
    start_time = time.time()
    print(f"--- Starting Data Preparation for Chunk: {args.input_file} ---")
    
    config = load_config()
    params = {**config['processing_parameters'], **config['training_parameters'], **config.get('data_processing', {})}
    
    project_root = config['project_root']
    # The script now looks for chunks inside the 'split_parts' subfolder
    prepared_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['prepared_subfolder'], 'split_parts')
    output_dl_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    os.makedirs(output_dl_folder, exist_ok=True)

    prepared_dataset_path = os.path.join(prepared_folder, args.input_file)
    print(f"\nProcessing specified dataset: {prepared_dataset_path}")
    if not os.path.exists(prepared_dataset_path):
        print(f"FATAL: Input chunk not found: {prepared_dataset_path}")
        return

    # For consistency, the scaler should be fitted once on a sample of the data (the first chunk).
    scaler_path = os.path.join(output_dl_folder, 'minmax_scaler.pkl')
    numerical_features = params.get('numerical_features', ['gc_content', 'dg', 'conservation'])
    if not os.path.exists(scaler_path):
        print("\nScaler not found. Fitting a new scaler on this first chunk...")
        try:
            numerical_df = pd.read_parquet(prepared_dataset_path, columns=numerical_features)
            scaler = MinMaxScaler()
            scaler.fit(numerical_df)
            joblib.dump(scaler, scaler_path)
            print(f"  - Scaler fitted and saved to {scaler_path}")
            del numerical_df
        except Exception as e:
            print(f"FATAL: Could not read numerical features to fit scaler: {e}")
            return
    else:
        scaler = joblib.load(scaler_path)
        print("\nLoaded existing scaler.")

    parquet_file = pq.ParquetFile(prepared_dataset_path)
    num_rows = parquet_file.metadata.num_rows
    indices = np.arange(num_rows)
    train_indices, test_indices = train_test_split(indices, test_size=params.get('test_split_ratio', 0.2), random_state=42)
    
    sample_assignment = np.empty(num_rows, dtype='U5')
    sample_assignment[train_indices] = 'train'
    sample_assignment[test_indices] = 'test'
    del indices, train_indices, test_indices

    # Define constants from config
    pad_params = params['sequence_padding']
    max_primary_len = pad_params.get('max_primary_len', 200)
    max_target_len = pad_params.get('max_target_len', 10000)
    max_competitor_len = pad_params.get('max_competitor_len', 10000)
    target_feature = params.get('target_feature', 'affinity')
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    
    def process_chunk_df(df_chunk, scaler_obj):
        y = df_chunk[target_feature].values.astype(np.float32)
        X = {
            'primary_sequence_input': np.array([one_hot_encode_sequence(seq, max_primary_len, nucleotide_map) for seq in df_chunk['primary_sequence']]),
            'target_sequence_input': np.array([one_hot_encode_sequence(seq, max_target_len, nucleotide_map) for seq in df_chunk['target_sequence']]),
            'competitor_sequence_input': np.array([one_hot_encode_sequence(seq, max_competitor_len, nucleotide_map) for seq in df_chunk['competitor_sequence']]),
            'primary_structure_input': np.expand_dims(pad_sequences(df_chunk['structure_vector'].apply(json.loads).tolist(), maxlen=max_primary_len, padding='post', dtype='float32'), axis=-1),
            'numerical_features_input': scaler_obj.transform(df_chunk[numerical_features])
        }
        return X, y

    batch_iterator = parquet_file.iter_batches(batch_size=params.get('batch_size_parquet', 5000))
    X_train_batches, y_train_batches = {key: [] for key in process_chunk_df(next(batch_iterator).to_pandas(), scaler)[0]}, []
    X_test_batches, y_test_batches = {key: [] for key in X_train_batches}, []
    
    batch_iterator = parquet_file.iter_batches(batch_size=params.get('batch_size_parquet', 5000))
    processed_rows = 0
    for batch in batch_iterator:
        df = batch.to_pandas()
        batch_indices = np.arange(processed_rows, processed_rows + len(df))
        assignments = sample_assignment[batch_indices]
        train_mask, test_mask = (assignments == 'train'), (assignments == 'test')

        if np.any(train_mask):
            X_train, y_train = process_chunk_df(df[train_mask], scaler)
            for key in X_train_batches: X_train_batches[key].append(X_train[key])
            y_train_batches.append(y_train)

        if np.any(test_mask):
            X_test, y_test = process_chunk_df(df[test_mask], scaler)
            for key in X_test_batches: X_test_batches[key].append(X_test[key])
            y_test_batches.append(y_test)
            
        processed_rows += len(df)
        print(f"  - Processed {processed_rows}/{num_rows} rows...", end='\r')

    print("\n\nStep 4: Concatenating and saving final NumPy arrays...")
    output_suffix = args.output_suffix

    if y_train_batches:
        y_train_final = np.concatenate(y_train_batches)
        np.save(os.path.join(output_dl_folder, f'y_train{output_suffix}.npy'), y_train_final)
        for key in X_train_batches:
            X_train_final = np.concatenate(X_train_batches[key])
            np.save(os.path.join(output_dl_folder, f'X_train_{key}{output_suffix}.npy'), X_train_final)
        print(f"  - Saved {len(y_train_final)} training samples to chunked files.")
    
    if y_test_batches:
        y_test_final = np.concatenate(y_test_batches)
        np.save(os.path.join(output_dl_folder, f'y_test{output_suffix}.npy'), y_test_final)
        for key in X_test_batches:
            X_test_final = np.concatenate(X_test_batches[key])
            np.save(os.path.join(output_dl_folder, f'X_test_{key}{output_suffix}.npy'), X_test_final)
        print(f"  - Saved {len(y_test_final)} test samples to chunked files.")
        
    end_time = time.time()
    print(f"\n--- Chunk Processing Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a single Parquet data chunk for Deep Learning.")
    parser.add_argument('--input_file', type=str, required=True, help='Filename of the Parquet chunk to process from the split_parts folder.')
    parser.add_argument('--output_suffix', type=str, required=True, help='A unique suffix for the output .npy files (e.g., _chunk1)')
    args = parser.parse_args()
    main(args)