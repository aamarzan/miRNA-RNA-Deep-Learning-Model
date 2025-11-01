# s0c_estimate_dataset_size.py
import os
import json
from Bio import SeqIO
import pandas as pd
import glob
import time


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

def format_time(seconds):
    """Formats seconds into a human-readable string (HH:MM:SS)."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

def calculate_final_dataset_size():
    """
    Performs a dry run calculation of the final dataset size and estimates
    the runtime for the entire pipeline.
    """
    print("--- Starting Dataset Size & Time Estimation Script ---")
    config = load_config()
    
    # --- 1. Load Paths and Parameters from Config ---
    project_root = config.get('project_root')
    raw_data_folder = os.path.join(project_root, 'dataset', 'raw_data')
    proc_params = config.get('processing_parameters', {})
    train_params = config.get('training_parameters', {})

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

    # --- 2. Calculate Balanced Primary Molecule Set ---
    print("\n--- 2. Calculating Size of Balanced Primary miRNA Set ---")
    mirna_ids = set(mirna_records.keys())
    known_mirna_ids = mirna_ids.intersection(affinity_ids)
    unknown_mirna_ids = mirna_ids - affinity_ids
    ratio = proc_params.get('downsampling_ratio_unknown_to_known', 1.0)
    num_known = len(known_mirna_ids)
    num_unknown_to_keep = min(len(unknown_mirna_ids), int(num_known * ratio))
    total_balanced_mirnas = num_known + num_unknown_to_keep
    
    print(f"  - Total Primary Molecules in Final Set: {total_balanced_mirnas}")

    # --- 3. Calculate Target Chunks ---
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
    else:
        total_target_chunks = len(target_records)
    print(f"  - Total Target Chunks to be Processed: {total_target_chunks}")

    # --- 4. Calculate Final Dataset Size ---
    print("\n--- 4. Calculating Final Dataset Size ---")
    num_competitors_augmented = len(competitor_records) + 1
    final_combinations = total_balanced_mirnas * total_target_chunks * num_competitors_augmented
    
    print(f"  - Balanced miRNAs: {total_balanced_mirnas}")
    print(f"  - Target Chunks: {total_target_chunks}")
    print(f"  - Competitors (incl. null): {num_competitors_augmented}")
    print("---------------------------------")
    print(f"  - Estimated Total Rows in Final Parquet File: {final_combinations:,}")
    print("---------------------------------")

    # --- 5. Estimate Pipeline Runtime ---
    print("\n--- 5. Estimating Pipeline Runtime ---")
    # Performance Assumptions (seconds per item)
    TIME_PER_FEATURE_ENG = 0.2  # Time to run RNAfold, etc. per molecule
    TIME_PER_ROW_NPZ = 0.0001 # Time to convert one row from Parquet to NPZ
    TIME_PER_TRAIN_BATCH = 0.5 # Time to train one batch on a moderate GPU/CPU
    TIME_PER_EVAL_SAMPLE = 0.001 # Time to predict on one sample for evaluation

    # s1a Estimate
    total_molecules_to_process = total_balanced_mirnas + total_target_chunks + len(competitor_records)
    time_s1a = total_molecules_to_process * TIME_PER_FEATURE_ENG

    # s2a Estimate
    time_s2a = final_combinations * TIME_PER_ROW_NPZ
    
    # s3a Estimate
    epochs = train_params.get('epochs', 40)
    batch_size = train_params.get('batch_size', 256)
    test_split = train_params.get('test_split_ratio', 0.2)
    train_samples = final_combinations * (1 - test_split)
    batches_per_epoch = train_samples / batch_size
    time_s3a = epochs * batches_per_epoch * TIME_PER_TRAIN_BATCH

    # s5a Estimate
    test_samples = final_combinations * test_split
    time_s5a = test_samples * TIME_PER_EVAL_SAMPLE

    print(f"s1a (Feature Engineering): Estimated ~{format_time(time_s1a)}")
    print(f"s2a (NPZ Conversion):      Estimated ~{format_time(time_s2a)}")
    print(f"s3a (Model Training):      Estimated ~{format_time(time_s3a)}")
    print(f"s5a (Evaluation):          Estimated ~{format_time(time_s5a)}")
    print("---------------------------------")
    print(f"Total Estimated Time:      ~{format_time(time_s1a + time_s2a + time_s3a + time_s5a)}")
    print("---------------------------------")
    print("\nNOTE: These are rough estimates. Actual time will vary based on your specific hardware.")


if __name__ == "__main__":
    calculate_final_dataset_size()