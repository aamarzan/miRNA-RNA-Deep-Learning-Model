# s0c_estimate_dataset_size.py
import os
import json
from Bio import SeqIO
import pandas as pd
import glob

def load_config(config_path=None):
    """Loads and returns the configuration file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Configuration file not found at '{config_path}'.")
        exit()

def count_fasta_records(folder_path):
    """Counts all records in all FASTA files in a folder."""
    count = 0
    fasta_extensions = ('.fa', '.fasta', '.fna', '.txt')
    fasta_files = [f for f in os.listdir(folder_path) if f.lower().endswith(fasta_extensions)]
    for filename in fasta_files:
        filepath = os.path.join(folder_path, filename)
        count += len(list(SeqIO.parse(filepath, "fasta")))
    return count

def get_all_fasta_records(folder_path):
    """Loads all records from all FASTA files in a folder."""
    records = {}
    fasta_extensions = ('.fa', '.fasta', '.fna', '.txt')
    fasta_files = [f for f in os.listdir(folder_path) if f.lower().endswith(fasta_extensions)]
    for filename in fasta_files:
        filepath = os.path.join(folder_path, filename)
        for record in SeqIO.parse(filepath, "fasta"):
            records[record.id.strip().split()[0]] = str(record.seq)
    return records

def calculate_final_dataset_size():
    """
    Performs a dry run calculation of the final dataset size based on the config.
    """
    print("--- Starting Dataset Size Estimation Script ---")
    config = load_config()
    
    # --- 1. Load Paths and Parameters from Config ---
    project_root = config.get('project_root')
    raw_data_folder = os.path.join(project_root, 'dataset', 'raw_data')
    proc_params = config.get('processing_parameters', {})

    print("\n--- 1. Counting Molecules from Source Files ---")
    try:
        mirna_records = get_all_fasta_records(os.path.join(raw_data_folder, config['data_sources']['mirna']['folder']))
        target_records = get_all_fasta_records(os.path.join(raw_data_folder, config['data_sources']['rna_target']['folder']))
        competitor_records = get_all_fasta_records(os.path.join(raw_data_folder, config['data_sources']['protein_competitor']['folder']))
        
        print(f"  - Found {len(mirna_records)} total primary miRNA sequences.")
        print(f"  - Found {len(target_records)} total target sequences.")
        print(f"  - Found {len(competitor_records)} total competitor sequences.")

        affinity_folder = os.path.join(raw_data_folder, config['data_sources']['affinity']['folder'])
        affinity_files = glob.glob(os.path.join(affinity_folder, "*.txt")) + glob.glob(os.path.join(affinity_folder, "*.csv"))
        df_affinity = pd.concat([pd.read_csv(f, sep='\t' if f.lower().endswith(('.txt', '.tsv')) else ',') for f in affinity_files])
        affinity_ids = {str(val).strip().split()[0] for val in df_affinity[config['data_sources']['affinity']['id_col']]}
        
    except Exception as e:
        print(f"❌ ERROR: Could not load data files. {e}")
        return

    # --- 2. Calculate the Number of Balanced Primary Molecules ---
    print("\n--- 2. Calculating Size of Balanced Primary miRNA Set ---")
    mirna_ids = set(mirna_records.keys())
    known_mirna_ids = mirna_ids.intersection(affinity_ids)
    unknown_mirna_ids = mirna_ids - affinity_ids

    ratio = proc_params.get('downsampling_ratio_unknown_to_known', 1.0)
    num_known = len(known_mirna_ids)
    num_unknown_to_keep = min(len(unknown_mirna_ids), int(num_known * ratio))
    
    total_balanced_mirnas = num_known + num_unknown_to_keep
    
    print(f"  - Number of miRNAs with known affinity: {num_known}")
    print(f"  - Number of 'unknown' miRNAs to be sampled (Ratio ≈ 1:{ratio}): {num_unknown_to_keep}")
    print(f"  - Total Primary Molecules in Final Set: {total_balanced_mirnas}")

    # --- 3. Calculate the Number of Target Chunks (Sliding Window) ---
    print("\n--- 3. Calculating Number of Target Chunks ---")
    sw_params = proc_params.get('sliding_window', {})
    use_sw = sw_params.get('use_sliding_window', False)
    window_size = sw_params.get('window_size', 500)
    step_size = sw_params.get('step_size', 250)
    
    total_target_chunks = 0
    if use_sw:
        for seq in target_records.values():
            if len(seq) > window_size:
                num_chunks = (len(seq) - window_size) // step_size + 1
                total_target_chunks += num_chunks
            else:
                total_target_chunks += 1
        print(f"  - Sliding window is ON. The {len(target_records)} targets will be sliced into {total_target_chunks} chunks.")
    else:
        total_target_chunks = len(target_records)
        print(f"  - Sliding window is OFF. Total targets: {total_target_chunks}")

    # --- 4. Calculate Final Number of Combinations ---
    print("\n--- 4. Calculating Final Dataset Size ---")
    num_competitors_augmented = len(competitor_records) + 1 # Add 1 for the null competitor
    
    final_combinations = total_balanced_mirnas * total_target_chunks * num_competitors_augmented
    
    print("\n--- Final Estimate ---")
    print(f"Balanced Primary miRNAs: {total_balanced_mirnas}")
    print(f"Total Target Chunks: {total_target_chunks}")
    print(f"Total Competitors (incl. null): {num_competitors_augmented}")
    print("---------------------------------")
    print(f"Estimated Total Rows in Final Parquet File: {final_combinations:,}")
    print("---------------------------------")


if __name__ == "__main__":
    calculate_final_dataset_size()