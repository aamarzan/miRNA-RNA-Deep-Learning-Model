# s2b_merge_npy_chunks.py
import os
import numpy as np
import json
from natsort import natsorted

def main():
    print("--- Starting .npy Chunk Merging Process ---")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        data_path = os.path.join(config['project_root'], config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    except Exception as e:
        print(f"FATAL: Could not load config.json. Error: {e}")
        return
        
    print(f"Searching for chunked files in: {data_path}")
    
    all_files = os.listdir(data_path)
    file_keys = set()
    for f in all_files:
        if '_chunk' in f and f.endswith('.npy'):
            base_name = f.split('_chunk')[0]
            file_keys.add(base_name)
    
    if not file_keys:
        print("No chunked files found to merge.")
        return
        
    print(f"Found {len(file_keys)} file types. Starting merge...")

    for key in sorted(list(file_keys)):
        print(f"  - Merging '{key}'...")
        chunk_files = natsorted([f for f in all_files if f.startswith(key) and '_chunk' in f])
        
        chunks_to_merge = []
        for chunk_filename in chunk_files:
            chunks_to_merge.append(np.load(os.path.join(data_path, chunk_filename)))
        
        if chunks_to_merge:
            final_array = np.concatenate(chunks_to_merge, axis=0)
            final_filename = os.path.join(data_path, f"{key}.npy")
            np.save(final_filename, final_array)
            print(f"    - Saved merged file: {final_filename} with shape {final_array.shape}")

    print("\n--- Merging Complete ---")

if __name__ == "__main__":
    main()