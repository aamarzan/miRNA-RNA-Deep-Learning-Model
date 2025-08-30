# s0_check_setup.py (Updated to read all FASTA files)
import os
import json
import pandas as pd
from Bio import SeqIO

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

def check_path(path, description, is_file=False):
    """Checks if a path exists and prints a formatted status message."""
    exists = os.path.isfile(path) if is_file else os.path.isdir(path)
    status = "✅ OK" if exists else "❌ ERROR: Not Found"
    print(f"{description:<50} | Status: {status}")
    return exists

def run_id_matching_diagnostics(mirna_folder, affinity_folder, affinity_cols):
    """
    Performs a comprehensive ID matching check across ALL files in the
    miRNA and affinity source folders.
    """
    print("\n--- Running Comprehensive ID Matching Diagnostics ---")
    try:
        def normalize_id(key):
            return str(key).strip().split()[0]

        # Load IDs from ALL FASTA files in the miRNA folder
        mirna_ids = set()
        fasta_extensions = ('.fa', '.fasta', '.fna', '.txt')
        mirna_files = [f for f in os.listdir(mirna_folder) if f.lower().endswith(fasta_extensions)]
        if not mirna_files: raise FileNotFoundError("No FASTA files found in miRNA folder.")
        
        for filename in mirna_files:
            filepath = os.path.join(mirna_folder, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                mirna_ids.add(normalize_id(record.id))
        
        # --- FIX: Load IDs from ALL score files in the affinity folder ---
        affinity_ids = set()
        score_extensions = ('.csv', '.tsv', '.txt')
        affinity_files = [f for f in os.listdir(affinity_folder) if f.lower().endswith(score_extensions)]
        if not affinity_files: raise FileNotFoundError("No score files (csv, tsv, txt) found in affinity folder.")

        for filename in affinity_files:
            filepath = os.path.join(affinity_folder, filename)
            sep = '\t' if filepath.lower().endswith(('.tsv', '.txt')) else ','
            df = pd.read_csv(filepath, sep=sep, comment='#', usecols=[affinity_cols['id_col']], low_memory=False)
            affinity_ids.update({normalize_id(val) for val in df[affinity_cols['id_col']].dropna()})

        print("\n--- Sample IDs Found ---")
        print(f"Sample IDs from all miRNA FASTA files combined:")
        print(f"  {list(mirna_ids)[:10]}")
        print(f"\nSample IDs from all Affinity files combined:")
        print(f"  {list(affinity_ids)[:10]}")
        print("------------------------\n")

        matched_ids = mirna_ids.intersection(affinity_ids)

        print(f"Total Unique miRNA IDs from FASTA (normalized): {len(mirna_ids)}")
        print(f"Total Unique IDs from Affinity File (normalized): {len(affinity_ids)}")
        print(f"  - Number of Matched IDs: {len(matched_ids)}")

        if not matched_ids:
            print("  - ❌ CRITICAL WARNING: Zero IDs matched between your miRNA FASTA and affinity score files!")
        
    except Exception as e:
        print(f"  - ❌ ERROR during ID check: {e}")

def main():
    """Runs a series of fast checks on the project setup."""
    print("--- Starting Project Setup Diagnostic Script ---\n")
    config = load_config()
    
    project_root = config.get('project_root')
    if not project_root:
        print("❌ FATAL ERROR: 'project_root' is not defined in config.json.")
        return

    # 1. Check Core Directories and Folders
    print("--- Checking Directories and Folders ---")
    data_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'])
    check_path(data_folder, "Main 'dataset' folder")
    raw_data_folder = os.path.join(data_folder, 'raw_data')
    check_path(raw_data_folder, "'raw_data' subfolder")

    for name, source_info in config['data_sources'].items():
        path = os.path.join(raw_data_folder, source_info['folder'])
        check_path(path, f"Source folder for '{name}'")
        
    # 2. Perform Comprehensive ID Matching Diagnostics
    mirna_folder = os.path.join(raw_data_folder, config['data_sources']['mirna']['folder'])
    affinity_folder = os.path.join(raw_data_folder, config['data_sources']['affinity']['folder'])
    affinity_cols = config['data_sources']['affinity']
    run_id_matching_diagnostics(mirna_folder, affinity_folder, affinity_cols)
        
    # 3. Check Config Logic
    print("\n--- Checking Config File Logic ---")
    proc_params = config.get('processing_parameters', {})
    train_params = config.get('training_parameters', {})
    sw_params = proc_params.get('sliding_window', {})
    pad_params = train_params.get('sequence_padding', {})

    if sw_params.get('use_sliding_window'):
        if sw_params.get('window_size') != pad_params.get('max_target_len'):
            print("  - ❌ CONFIG ERROR: 'window_size' does not match 'max_target_len' in 'sequence_padding'.")
        else:
            print("  - ✅ OK: Sliding window size matches model padding length.")
    else:
        print("  - ✅ OK: Sliding window is disabled.")

    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    main()