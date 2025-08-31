# s0b_check_conservation.py
# PURPOSE:
# A fast-running diagnostic script to check the match rate between the miRNA
# families found in the main FASTA dataset and the families in the processed
# conservation score file.
#
import os
import json
import pandas as pd
from Bio import SeqIO
import re

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

def extract_family_from_id(mirna_id):
    """
    Extracts the core family name from a miRNA ID using regex.
    e.g., 'hsa-let-7a-1' -> 'let-7'
    e.g., 'mmu-mir-21a-5p' -> 'mir-21'
    """
    # This regex is designed to find common miRNA family patterns like 'mir-#' or 'let-#'
    match = re.search(r"(mir-|let-)\d+[a-z]?", mirna_id.lower())
    if match:
        return match.group(0)
    return None

def main():
    """Runs the diagnostic check for conservation score matching."""
    print("--- Starting Conservation Score Match Diagnostic ---\n")
    config = load_config()
    
    project_root = config.get('project_root')
    if not project_root:
        print("❌ FATAL ERROR: 'project_root' is not defined in config.json.")
        return

    # --- 1. Define Paths ---
    raw_data_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'], 'raw_data')
    mirna_folder = os.path.join(raw_data_folder, config['data_sources']['mirna']['folder'])
    conservation_folder = os.path.join(raw_data_folder, config['data_sources']['conservation']['folder'])
    
    # --- 2. Extract Unique Family Names from all miRNA FASTA files ---
    print(f"🔎 Scanning all FASTA files in: '{mirna_folder}'...")
    mirna_families_from_fasta = set()
    try:
        fasta_extensions = ('.fa', '.fasta', '.fna', '.txt')
        mirna_files = [f for f in os.listdir(mirna_folder) if f.lower().endswith(fasta_extensions)]
        if not mirna_files: raise FileNotFoundError("No FASTA files found.")
        
        for filename in mirna_files:
            filepath = os.path.join(mirna_folder, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                family = extract_family_from_id(record.id)
                if family:
                    mirna_families_from_fasta.add(family)
        print(f"  - Found {len(mirna_families_from_fasta)} unique miRNA families in your FASTA files.")
    except Exception as e:
        print(f"  - ❌ ERROR reading miRNA files: {e}")
        return

    # --- 3. Extract Unique Family Names from Conservation Score File ---
    print(f"\n🔎 Scanning all score files in: '{conservation_folder}'...")
    families_from_conservation_file = set()
    try:
        score_extensions = ('.csv', '.tsv', '.txt')
        conservation_files = [f for f in os.listdir(conservation_folder) if f.lower().endswith(score_extensions)]
        if not conservation_files: raise FileNotFoundError("No conservation score files found.")
        
        id_col = config['data_sources']['conservation']['id_col']
        
        for filename in conservation_files:
            filepath = os.path.join(conservation_folder, filename)
            sep = '\t' if filepath.lower().endswith(('.tsv', '.txt')) else ','
            df = pd.read_csv(filepath, sep=sep, comment='#', usecols=[id_col], low_memory=False)
            
            # Normalize the family names from the file (e.g., to lowercase)
            families_from_conservation_file.update({str(val).lower() for val in df[id_col].dropna()})
        print(f"  - Found {len(families_from_conservation_file)} unique miRNA families in your conservation file(s).")
    except Exception as e:
        print(f"  - ❌ ERROR reading conservation files: {e}")
        return
        
    # --- 4. Compare and Report ---
    print("\n--- Conservation Match Report ---")
    
    matched_families = mirna_families_from_fasta.intersection(families_from_conservation_file)
    unmatched_families = mirna_families_from_fasta - families_from_conservation_file

    if mirna_families_from_fasta:
        match_rate = len(matched_families) / len(mirna_families_from_fasta)
        print(f"Total unique families in your FASTA data: {len(mirna_families_from_fasta)}")
        print(f"Matched families with conservation score: {len(matched_families)}")
        print(f"  - ✅ Match Rate: {match_rate:.2%}")

        if unmatched_families:
            print(f"\n  - ⚠️ NOTE: {len(unmatched_families)} families in your data have no conservation score.")
            print(f"    (These will be assigned a default score of 0.0. Example unmatched families: {list(unmatched_families)[:10]})")
    else:
        print("  - Could not find any miRNA families to analyze in your FASTA files.")

    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    main()