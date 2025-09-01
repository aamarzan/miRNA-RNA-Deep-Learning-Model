# s0_check_setup.py (UPGRADED VERSION 2)
# PURPOSE:
# A fast-running diagnostic script to validate project setup. It now simulates
# the downsampling of unknown miRNAs to predict the final dataset's statistics.
#
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

def run_data_diagnostics(mirna_folder, affinity_folder, affinity_cols, config):
    """
    A unified function that performs comprehensive ID matching and calculates
    statistics for ALL scores, MATCHED scores, and the FINAL SIMULATED dataset.
    """
    print("\n--- Running Combined Data Diagnostics (ID Matching & Statistics) ---")
    try:
        def normalize_id(key):
            return str(key).strip().split()[0]

        # 1. Load all miRNA IDs from FASTA files
        print("  - Loading miRNA IDs from all FASTA files...")
        mirna_ids = set()
        fasta_extensions = ('.fa', '.fasta', '.fna', '.txt')
        mirna_files = [f for f in os.listdir(mirna_folder) if f.lower().endswith(fasta_extensions)]
        if not mirna_files: raise FileNotFoundError("No FASTA files found in miRNA folder.")
        
        for filename in mirna_files:
            filepath = os.path.join(mirna_folder, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                mirna_ids.add(normalize_id(record.id))
        
        # 2. Load all affinity data into a dictionary {ID: score}
        print("  - Loading affinity IDs and scores from all score files...")
        affinity_data = {}
        score_extensions = ('.csv', '.tsv', '.txt')
        affinity_files = [f for f in os.listdir(affinity_folder) if f.lower().endswith(score_extensions)]
        if not affinity_files: raise FileNotFoundError("No score files (csv, tsv, txt) found in affinity folder.")

        for filename in affinity_files:
            filepath = os.path.join(affinity_folder, filename)
            sep = '\t' if filepath.lower().endswith(('.tsv', '.txt')) else ','
            
            df = pd.read_csv(filepath, sep=sep, comment='#', low_memory=False)
            id_col_expected, score_col_expected = affinity_cols['id_col'], affinity_cols['score_col']
            
            id_col_found = id_col_expected if id_col_expected in df.columns else next((alt for alt in ['miRNA', 'miRNA_ID'] if alt in df.columns), None)
            score_col_found = score_col_expected if score_col_expected in df.columns else next((alt for alt in ['microt_score', 'Affinity', 'Score'] if alt in df.columns), None)
            
            if not id_col_found or not score_col_found:
                raise ValueError(f"Could not find required ID or Score columns in {filename}.")

            for _, row in df.iterrows():
                norm_id = normalize_id(row[id_col_found])
                score = pd.to_numeric(row[score_col_found], errors='coerce')
                if pd.notna(score):
                    affinity_data[norm_id] = score

        # 3. Perform ID Matching Analysis
        affinity_ids = set(affinity_data.keys())
        matched_ids = mirna_ids.intersection(affinity_ids)

        print("\n--- ID Matching Summary ---")
        print(f"Total Unique miRNA IDs from FASTA (normalized): {len(mirna_ids)}")
        print(f"Total Unique IDs from Affinity File (normalized): {len(affinity_ids)}")
        print(f"  - ✅ Number of Matched IDs: {len(matched_ids)}")

        if not matched_ids:
            print("  - ❌ CRITICAL WARNING: Zero IDs matched between your miRNA FASTA and affinity score files!")
        
        # 4. Perform Statistical Analysis
        print("\n--- Affinity Score Statistics ---")
        if affinity_data:
            all_scores = pd.Series(list(affinity_data.values()))
            print("\n📊 Descriptive Statistics for ALL Affinity Scores:")
            print(all_scores.describe().to_string())
        else:
            print("No affinity scores were loaded.")

        if matched_ids:
            matched_scores_list = [affinity_data[mid] for mid in matched_ids]
            matched_scores = pd.Series(matched_scores_list)
            print("\n📊 Descriptive Statistics for MATCHED Affinity Scores Only:")
            print(matched_scores.describe().to_string())
        else:
            print("\nNo matched scores to analyze.")
        
        # 5. ⭐ NEW: Simulate the downsampling from s1a to predict final dataset statistics
        print("\n--- Statistics After Simulating Downsampling ---")
        unknown_ids = mirna_ids - affinity_ids
        ratio = config.get('processing_parameters', {}).get('downsampling_ratio_unknown_to_known', 1.0)
        num_known = len(matched_ids)
        num_unknown_to_keep = min(len(unknown_ids), int(num_known * ratio))
        
        print(f"  - Simulating the addition of {num_unknown_to_keep} 'unknown' miRNAs (with affinity 0.0)")
        print(f"  - Based on {num_known} known miRNAs and a ratio of {ratio:.2f} from config.json")

        if matched_ids:
            # Unknowns are treated as having 0.0 affinity
            unknown_scores_to_add = [0.0] * num_unknown_to_keep
            final_simulated_scores = pd.Series(matched_scores_list + unknown_scores_to_add)
            
            print("\n📊 Descriptive Statistics for Final SIMULATED Dataset:")
            print(final_simulated_scores.describe().to_string())
        else:
            print("\nCannot simulate downsampling without any matched miRNAs.")

    except Exception as e:
        print(f"  - ❌ ERROR during diagnostics: {e}")


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
            
    # 2. Perform Unified Diagnostics on Data Files
    mirna_folder = os.path.join(raw_data_folder, config['data_sources']['mirna']['folder'])
    affinity_folder = os.path.join(raw_data_folder, config['data_sources']['affinity']['folder'])
    affinity_cols = config['data_sources']['affinity']
    
    # Call the new unified function, passing the config object
    run_data_diagnostics(mirna_folder, affinity_folder, affinity_cols, config)
        
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