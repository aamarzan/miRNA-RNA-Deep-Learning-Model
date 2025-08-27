# s2b_merge_chunks.py (Final Version with controllable chunk number)
import os
import numpy as np
import json
import shutil
from natsort import natsorted

# --- USER CONFIGURATION ---
# Set this to the number of chunks you want to merge in this run.
# For your test, we will set it to 5.
NUMBER_OF_CHUNKS_TO_MERGE = 5
# --- END OF CONFIGURATION ---

def load_config(config_path=None):
    """Loads the configuration from a JSON file."""
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

def main():
    """
    Finds and merges a specified number of processed chunk folders.
    """
    print("--- Starting Controllable .npz Chunk Merging Process ---")
    
    config = load_config()
    
    # --- 1. Define Paths ---
    dataset_folder = os.path.join(config['project_root'], config['data_folders']['main_dataset_folder'])
    base_output_name = config['data_folders']['processed_for_dl_subfolder']
    final_output_folder = os.path.join(dataset_folder, base_output_name)
    os.makedirs(final_output_folder, exist_ok=True)
        
    print(f"Searching for chunk folders in: {dataset_folder}")
    
    # --- 2. Find all available chunk directories ---
    all_chunk_dirs = natsorted([d for d in os.listdir(dataset_folder) if d.startswith(base_output_name + '_') and os.path.isdir(os.path.join(dataset_folder, d))])
    
    if not all_chunk_dirs:
        print("No chunk folders found to merge.")
        return
        
    # <<< CHANGE: Select only the number of chunks you specified at the top >>>
    if NUMBER_OF_CHUNKS_TO_MERGE > len(all_chunk_dirs):
        print(f"Warning: You asked to merge {NUMBER_OF_CHUNKS_TO_MERGE} chunks, but only {len(all_chunk_dirs)} were found. Merging all available chunks.")
        dirs_to_process = all_chunk_dirs
    else:
        dirs_to_process = all_chunk_dirs[:NUMBER_OF_CHUNKS_TO_MERGE]

    print(f"Found {len(all_chunk_dirs)} total chunks. This run will merge the first {len(dirs_to_process)}: {dirs_to_process}")

    # --- 3. Identify file keys from the first chunk ---
    sample_dir = os.path.join(dataset_folder, dirs_to_process[0])
    file_keys = [f.replace('.npz', '') for f in os.listdir(sample_dir) if f.endswith('.npz')]

    # --- 4. Loop and merge ---
    for key in sorted(file_keys):
        print(f"  - Merging '{key}'...")
        chunks_to_merge = []
        for chunk_dir in dirs_to_process:
            chunk_path = os.path.join(dataset_folder, chunk_dir, f"{key}.npz")
            if os.path.exists(chunk_path):
                with np.load(chunk_path) as loaded_file:
                    chunks_to_merge.append(loaded_file['data'])
        
        if chunks_to_merge:
            final_array = np.concatenate(chunks_to_merge, axis=0)
            final_filename = os.path.join(final_output_folder, f"{key}.npz")
            np.savez_compressed(final_filename, data=final_array)
            print(f"    - Saved merged file: {final_filename} with shape {final_array.shape}")
    
    print("\n--- Merging Complete ---")
    print(f"Final processed data for the first {len(dirs_to_process)} chunks is ready in: {final_output_folder}")

if __name__ == "__main__":
    main()