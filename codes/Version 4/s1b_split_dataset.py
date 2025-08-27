# 1b_split_dataset.py (Fully Config-Driven, Memory-Safe Version)
import os
import pandas as pd
import pyarrow.parquet as pq
import math
import json

# <<< CHANGE: No more hard-coded configuration here. >>>

# --- Configuration Loader ---
def load_config(config_path=None):
    """Loads the configuration from a JSON file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
    
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

def split_dataset_memory_safe():
    """
    Splits a large Parquet file into smaller parts based on settings
    in the config.json file.
    """
    print("--- Starting Memory-Safe Dataset Dissection Script ---")

    # --- 1. Load Config and Define Paths ---
    config = load_config()
    split_params = config.get('dataset_splitting', {})
    
    project_root = config['project_root']
    prepared_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['prepared_subfolder'])
    output_folder = os.path.join(prepared_folder, "split_parts")
    os.makedirs(output_folder, exist_ok=True)

    # --- 2. Determine Input File ---
    input_filename = split_params.get('input_filename', '')
    if not input_filename:
        # Auto-detect the most recent Parquet file if none is specified
        try:
            prepared_files = [f for f in os.listdir(prepared_folder) if f.endswith('.parquet') and os.path.isfile(os.path.join(prepared_folder, f))]
            if not prepared_files: raise FileNotFoundError("No Parquet files found.")
            input_filename = sorted(prepared_files)[-1]
            print(f"  - No input file specified. Auto-detecting latest dataset: {input_filename}")
        except (FileNotFoundError, IndexError) as e:
            print(f"\nFATAL ERROR in auto-detection: {e}. Please run Stage 1 first.")
            return
    
    input_path = os.path.join(prepared_folder, input_filename)
    
    print(f"Input file: {input_path}")
    print(f"Output folder for split parts: {output_folder}")

    if not os.path.exists(input_path):
        print(f"\nFATAL ERROR: Input file not found at '{input_path}'.")
        return

    # --- 3. Open Parquet File and Plan the Split ---
    try:
        parquet_file = pq.ParquetFile(input_path)
        total_row_groups = parquet_file.num_row_groups
        number_of_parts = split_params.get('number_of_parts', 10)
        
        if total_row_groups < number_of_parts:
            print(f"\nWarning: The number of parts ({number_of_parts}) is greater than the number of chunks in the file ({total_row_groups}).")
            print(f"         The script will create {total_row_groups} smaller files instead.")
            num_parts_to_create = total_row_groups
        else:
            num_parts_to_create = number_of_parts
            
        if num_parts_to_create > 0:
            groups_per_part = math.ceil(total_row_groups / num_parts_to_create)
        else:
            groups_per_part = 0
        
        print(f"  - Input file has {total_row_groups} internal chunks (row groups).")
        print(f"  - Each of the {num_parts_to_create} output files will contain up to {groups_per_part} chunks.")

    except Exception as e:
        print(f"  - Error reading Parquet file: {e}")
        return

    # --- 4. Read Chunks and Write to New Files ---
    print("\nSplitting the dataset and saving parts...")
    
    current_group_index = 0
    for part_num in range(1, num_parts_to_create + 1):
        output_filename = f"{os.path.splitext(input_filename)[0]}_part_{part_num}_of_{num_parts_to_create}.parquet"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            start_group = current_group_index
            end_group = min(current_group_index + groups_per_part, total_row_groups)

            with pq.ParquetWriter(output_path, parquet_file.schema_arrow) as writer:
                for i in range(start_group, end_group):
                    writer.write_table(parquet_file.read_row_group(i))
            
            print(f"  - Successfully saved '{output_filename}' (contains chunks {start_group}-{end_group-1}).")
            current_group_index = end_group

        except Exception as e:
            print(f"  - Error saving part {part_num}: {e}")

    print("\n--- Dataset Dissection Complete ---")
    print(f"All parts have been saved to: '{output_folder}'")

if __name__ == "__main__":
    split_dataset_memory_safe()