# s2b_merge_chunks.py (Full, Final Version)
import os
import numpy as np
import json
import shutil
from natsort import natsorted

def load_config(config_path=None):
    """
    Loads the configuration from a JSON file.
    """
    if config_path is None:
        # Assumes the script is in a /codes folder and config.json is one level up
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
    Finds all processed chunk folders, merges their .npz files into final
    .npz files, and cleans up the temporary chunk folders.
    """
    print("--- Starting .npz Chunk Merging Process from Separate Folders ---")
    
    config = load_config()
    
    # --- 1. Define Paths ---
    dataset_folder = os.path.join(config['project_root'], config['data_folders']['main_dataset_folder'])
    base_output_name = config['data_folders']['processed_for_dl_subfolder']
    final_output_folder = os.path.join(dataset_folder, base_output_name)
    os.makedirs(final_output_folder, exist_ok=True)
        
    print(f"Searching for chunk folders matching '{base_output_name}_*' in: {dataset_folder}")
    
    # --- 2. Find all chunk directories ---
    # e.g., 'processed_for_dl_1', 'processed_for_dl_2'
    chunk_dirs = natsorted([d for d in os.listdir(dataset_folder) if d.startswith(base_output_name + '_') and os.path.isdir(os.path.join(dataset_folder, d))])
    
    if not chunk_dirs:
        print("No chunk folders found to merge. Please run the s2 processing script first.")
        return
        
    print(f"Found {len(chunk_dirs)} chunk folders: {chunk_dirs}.")
    print("Identifying file types from the first chunk...")

    # --- 3. Identify all unique file keys from the first chunk ---
    try:
        sample_dir = os.path.join(dataset_folder, chunk_dirs[0])
        file_keys = [f.replace('.npz', '') for f in os.listdir(sample_dir) if f.endswith('.npz')]
    except (IndexError, FileNotFoundError):
        print(f"Error: The first chunk folder '{chunk_dirs[0]}' seems to be empty or missing.")
        return

    # --- 4. Loop through each file type and merge its chunks ---
    for key in sorted(file_keys):
        print(f"  - Merging '{key}'...")
        chunks_to_merge = []
        for chunk_dir in chunk_dirs:
            chunk_path = os.path.join(dataset_folder, chunk_dir, f"{key}.npz")
            if os.path.exists(chunk_path):
                try:
                    with np.load(chunk_path) as loaded_file:
                        chunks_to_merge.append(loaded_file['data'])
                except Exception as e:
                    print(f"    - WARNING: Could not load or read {chunk_path}. Skipping. Error: {e}")
        
        if chunks_to_merge:
            final_array = np.concatenate(chunks_to_merge, axis=0)
            final_filename = os.path.join(final_output_folder, f"{key}.npz")
            np.savez_compressed(final_filename, data=final_array)
            print(f"    - Saved final merged file: {final_filename} with shape {final_array.shape}")

    # --- 5. Clean up temporary chunk folders ---
    print("\nCleaning up temporary chunk folders...")
    for chunk_dir in chunk_dirs:
        try:
            shutil.rmtree(os.path.join(dataset_folder, chunk_dir))
            print(f"  - Removed folder: {chunk_dir}")
        except Exception as e:
            print(f"  - WARNING: Could not remove folder {chunk_dir}. Error: {e}")
    
    print("\n--- Merging Complete ---")
    print(f"Final processed data is ready in: {final_output_folder}")

if __name__ == "__main__":
    main()