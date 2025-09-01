# s0c_check_sequence_lengths.py (UPGRADED VERSION)
# PURPOSE:
# To analyze the raw FASTA files for miRNAs, targets, and competitors, and calculate
# detailed descriptive statistics (mean, median, quartiles, etc.) for their
# effective nucleotide lengths. This helps in choosing optimal 'max_len' parameters.

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

def detect_sequence_type(sequence):
    """
    Detects if a sequence is RNA or Protein based on its alphabet.
    """
    # A simplified but effective check for protein-specific amino acids
    protein_specific_alphabet = set("LIVFWYMPSTHRKQNDE")
    seq_set = set(str(sequence).upper())
    
    if seq_set.intersection(protein_specific_alphabet):
        return "protein"
    return "rna"

def get_length_statistics(folder_path):
    """
    Calculates descriptive statistics for effective nucleotide lengths for all FASTA files in a folder.
    """
    all_lengths = []
    
    fasta_extensions = ('.fa', '.fasta', '.fna', '.txt')
    fasta_files = [f for f in os.listdir(folder_path) if f.lower().endswith(fasta_extensions)]
    
    if not fasta_files:
        print(f"  - No FASTA files found in {os.path.basename(folder_path)} folder.")
        return None

    for filename in fasta_files:
        filepath = os.path.join(folder_path, filename)
        for record in SeqIO.parse(filepath, "fasta"):
            seq = record.seq
            seq_type = detect_sequence_type(seq)
            
            # Calculate the effective nucleotide length
            if seq_type == "protein":
                effective_length = len(seq) * 3
            else: # rna
                effective_length = len(seq)
            
            all_lengths.append(effective_length)

    if not all_lengths:
        return None

    # Use Pandas to calculate all statistics at once
    lengths_series = pd.Series(all_lengths)
    return lengths_series.describe()

if __name__ == "__main__":
    print("--- Analyzing Sequence Lengths from Raw Data ---")
    config = load_config()
    
    project_root = config.get('project_root')
    raw_data_folder = os.path.join(project_root, 'dataset', 'raw_data')
    
    # Get paths for ALL THREE molecule types from config
    mirna_folder = os.path.join(raw_data_folder, config['data_sources']['mirna']['folder'])
    target_folder = os.path.join(raw_data_folder, config['data_sources']['rna_target']['folder'])
    competitor_folder = os.path.join(raw_data_folder, config['data_sources']['protein_competitor']['folder'])
    
    print("\nAnalyzing miRNA sequences...")
    mirna_stats = get_length_statistics(mirna_folder)
    
    print("\nAnalyzing Target sequences...")
    target_stats = get_length_statistics(target_folder)
    
    print("\nAnalyzing Competitor sequences...")
    competitor_stats = get_length_statistics(competitor_folder)
    
    print("\n--- Sequence Length Summary ---")
    print("Note: Amino acid sequence lengths have been converted to nucleotide lengths (x3).")
    
    print("\n📊 miRNAs:")
    if mirna_stats is not None:
        print(mirna_stats.to_string())
    else:
        print("  - No miRNA sequences found to analyze.")

    print("\n📊 Targets:")
    if target_stats is not None:
        print(target_stats.to_string())
    else:
        print("  - No Target sequences found to analyze.")

    print("\n📊 Competitors:")
    if competitor_stats is not None:
        print(competitor_stats.to_string())
    else:
        print("  - No Competitor sequences found to analyze.")
    
    print("\nUse these statistics (especially the 75% and max values) to make an informed decision on the 'max_len' settings.")