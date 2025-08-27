# s2_prepare_dl_data.py (Full, Final Version for Chunk Processing)
import os
import pandas as pd# convert_npy_to_npz.py
import os
import numpy as np

# --- USER CONFIGURATION ---
# IMPORTANT: Set this to the path of the folder containing your .npy files.
TARGET_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl" 
# --- END OF CONFIGURATION ---


def convert_folder_to_npz(folder_path):
    """
    Finds all .npy files in the specified folder, loads them, and saves them
    as compressed .npz files in the same location.
    """
    print(f"--- Starting conversion for folder: {folder_path} ---")

    # Check if the target folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return

    # Get a list of all files in the directory
    try:
        all_files = os.listdir(folder_path)
    except Exception as e:
        print(f"Error: Could not read folder contents. {e}")
        return
        
    npy_files = [f for f in all_files if f.endswith('.npy')]

    if not npy_files:
        print("No .npy files found in this folder.")
        return

    print(f"Found {len(npy_files)} .npy files to convert.")

    # Loop through each .npy file
    for npy_filename in npy_files:
        # Construct the full path for the input and output files
        npy_path = os.path.join(folder_path, npy_filename)
        base_name = os.path.splitext(npy_filename)[0]
        npz_filename = f"{base_name}.npz"
        npz_path = os.path.join(folder_path, npz_filename)

        try:
            # Load the data from the .npy file
            data_array = np.load(npy_path)

            # Save the data as a compressed .npz file
            # We save the array under the key 'data' for easy access later
            np.savez_compressed(npz_path, data=data_array)
            
            print(f"  - Successfully converted '{npy_filename}' -> '{npz_filename}'")

            # --- Optional: Delete the original .npy file ---
            # Uncomment the line below if you want to delete the old files after conversion.
            # WARNING: This is a permanent deletion.
            # os.remove(npy_path)
            # print(f"    - Deleted original file: '{npy_filename}'")

        except Exception as e:
            print(f"  - FAILED to convert '{npy_filename}'. Error: {e}")

    print("\n--- Conversion Complete ---")


if __name__ == "__main__":
    convert_folder_to_npz(TARGET_FOLDER)
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
    """
    Loads the configuration from a JSON file.
    If no path is given, it automatically finds 'config.json' in the project root.
    """
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
    """One-hot encodes a single sequence."""
    encoded_seq = np.zeros((max_len, len(nucleotide_map)), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, nucleotide_map.get(char.upper(), len(nucleotide_map) - 1)] = 1 # Default to 'N'
    return encoded_seq

# --- Main Processing Function ---
def main(args):
    """
    Processes a single Parquet chunk into compressed .npz files in a unique folder.
    """
    start_time = time.time()
    print(f"--- Starting Data Preparation for Chunk: {args.input_file} ---")
    
    config = load_config()
    params = {**config.get('processing_parameters', {}), **config.get('training_parameters', {}), **config.get('data_processing', {})}
    
    # --- 1. Define Paths ---
    project_root = config['project_root']
    dataset_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'])
    prepared_folder = os.path.join(dataset_folder, config['data_folders']['prepared_subfolder'], 'split_parts')
    
    # Create a unique main output folder for this chunk
    base_output_name = config['data_folders']['processed_for_dl_subfolder']
    chunk_output_folder_name = f"{base_output_name}{args.output_suffix}" # e.g., "processed_for_dl_1"
    output_dl_folder = os.path.join(dataset_folder, chunk_output_folder_name)
    os.makedirs(output_dl_folder, exist_ok=True)

    prepared_dataset_path = os.path.join(prepared_folder, args.input_file)
    if not os.path.exists(prepared_dataset_path):
        print(f"FATAL: Input chunk not found: {prepared_dataset_path}")
        return

    # --- 2. Handle Scaler ---
    # The scaler is created once (on chunk 1) and then re-used for all other chunks.
    # It's stored in the main processed folder, not the chunk-specific one.
    main_processed_folder = os.path.join(dataset_folder, base_output_name)
    os.makedirs(main_processed_folder, exist_ok=True)
    scaler_path = os.path.join(main_processed_folder, 'minmax_scaler.pkl')
    numerical_features = params.get('numerical_features', [])
    
    if not os.path.exists(scaler_path) and numerical_features:
        print(f"\nScaler not found at {scaler_path}. Fitting a new scaler on this first chunk...")
        try:
            numerical_df = pd.read_parquet(prepared_dataset_path, columns=numerical_features)
            scaler = MinMaxScaler()
            scaler.fit(numerical_df)
            joblib.dump(scaler, scaler_path)
            print(f"  - Scaler fitted and saved.")
            del numerical_df
        except Exception as e:
            print(f"FATAL: Could not read numerical features to fit scaler: {e}")
            return
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    if scaler:
        print("\nLoaded existing scaler.")
    else:
        print("\nNo numerical features to scale.")

    # --- 3. Prepare for Processing ---
    parquet_file = pq.ParquetFile(prepared_dataset_path)
    num_rows = parquet_file.metadata.num_rows
    indices = np.arange(num_rows)
    train_indices, test_indices = train_test_split(indices, test_size=params.get('test_split_ratio', 0.2), random_state=42)
    
    sample_assignment = np.empty(num_rows, dtype='U5')
    sample_assignment[train_indices] = 'train'
    sample_assignment[test_indices] = 'test'
    del indices, train_indices, test_indices

    pad_params = params.get('sequence_padding', {})
    max_primary_len = pad_params.get('max_primary_len', 200)
    max_target_len = pad_params.get('max_target_len', 2500)
    max_competitor_len = pad_params.get('max_competitor_len', 2500)
    target_feature = params.get('target_feature', 'affinity')
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    
    all_columns = parquet_file.schema.names
    
    def process_chunk_df(df_chunk, scaler_obj):
        y = df_chunk[target_feature].values.astype(np.float32)
        X = {}
        # Dynamically create outputs based on columns present in the Parquet file
        if 'primary_sequence' in all_columns:
            X['primary_sequence_input'] = np.array([one_hot_encode_sequence(seq, max_primary_len, nucleotide_map) for seq in df_chunk['primary_sequence']])
        if 'target_sequence' in all_columns:
            X['target_sequence_input'] = np.array([one_hot_encode_sequence(seq, max_target_len, nucleotide_map) for seq in df_chunk['target_sequence']])
        if 'competitor_sequence' in all_columns:
            X['competitor_sequence_input'] = np.array([one_hot_encode_sequence(seq, max_competitor_len, nucleotide_map) for seq in df_chunk['competitor_sequence']])
        if 'structure_vector' in all_columns:
            X['primary_structure_input'] = np.expand_dims(pad_sequences(df_chunk['structure_vector'].apply(json.loads).tolist(), maxlen=max_primary_len, padding='post', dtype='float32'), axis=-1)
        if 'adjacency_matrix' in all_columns:
             # Add more adjacency matrix processing here if needed in the future
            pass
        if numerical_features and scaler_obj:
            X['numerical_features_input'] = scaler_obj.transform(df_chunk[numerical_features])
        return X, y

    # --- 4. Process and Save in Batches ---
    print("\nProcessing and saving datasets...")
    batch_iterator = parquet_file.iter_batches(batch_size=params.get('batch_size_parquet', 5000))
    
    sample_df = next(batch_iterator).to_pandas()
    sample_X, _ = process_chunk_df(sample_df, scaler)
    X_train_batches, y_train_batches = {key: [] for key in sample_X}, []
    X_test_batches, y_test_batches = {key: [] for key in sample_X}, []
    
    all_batches_df = [sample_df] + [b.to_pandas() for b in batch_iterator]
    processed_rows = 0
    for df in all_batches_df:
        batch_indices = np.arange(processed_rows, processed_rows + len(df))
        assignments = sample_assignment[batch_indices]
        train_mask = (assignments == 'train')
        test_mask = (assignments == 'test')

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

    # --- 5. Finalize and Save COMPRESSED Arrays ---
    print("\n\nFinalizing and saving chunked files...")
    if y_train_batches:
        y_train_final = np.concatenate(y_train_batches)
        np.savez_compressed(os.path.join(output_dl_folder, 'y_train.npz'), data=y_train_final)
        for key in X_train_batches:
            X_train_final = np.concatenate(X_train_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_train_{key}.npz'), data=X_train_final)
        print(f"  - Saved {len(y_train_final)} training samples to folder '{chunk_output_folder_name}'")
    
    if y_test_batches:
        y_test_final = np.concatenate(y_test_batches)
        np.savez_compressed(os.path.join(output_dl_folder, 'y_test.npz'), data=y_test_final)
        for key in X_test_batches:
            X_test_final = np.concatenate(X_test_batches[key])
            np.savez_compressed(os.path.join(output_dl_folder, f'X_test_{key}.npz'), data=X_test_final)
        print(f"  - Saved {len(y_test_final)} test samples to folder '{chunk_output_folder_name}'")
        
    end_time = time.time()
    print(f"\n--- Chunk Processing Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Parquet data chunk for Deep Learning.")
    parser.add_argument('--input_file', type=str, required=True, help='Filename of the Parquet chunk from the split_parts folder.')
    parser.add_argument('--output_suffix', type=str, required=True, help='A unique suffix for the output folder (e.g., _1)')
    args = parser.parse_args()
    main(args)