# s0_final_setup_check.py
# PURPOSE:
# A final, comprehensive diagnostic script to validate the entire project setup
# before launching the time-consuming s1a_prepare_dataset.py script.
#
import os
import json
import pandas as pd
from Bio import SeqIO
import re
import warnings
from Bio.PDB import PDBParser, MMCIFParser, PDBExceptions
from Bio.PDB.Polypeptide import protein_letters_3to1_extended as aa3to1
import glob

# --- Helper Functions ---

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

def load_all_fasta_data(folder_path):
    """Loads all FASTA files in a folder and returns a dict of {id: sequence}."""
    fasta_data = {}
    fasta_extensions = ('.fa', '.fasta', '.fna', '.txt')
    fasta_files = [f for f in os.listdir(folder_path) if f.lower().endswith(fasta_extensions)]
    if not fasta_files: raise FileNotFoundError(f"No FASTA files found in {folder_path}")
    
    for filename in fasta_files:
        filepath = os.path.join(folder_path, filename)
        for record in SeqIO.parse(filepath, "fasta"):
            fasta_data[record.id.strip().split()[0]] = str(record.seq)
    return fasta_data

def get_sequence_from_pdb(pdb_path):
    """
    Extracts the sequence from a PDB or mmCIF file, handling both
    proteins and nucleic acids using Biopython's internal dictionaries.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBExceptions.PDBConstructionWarning)
            file_ext = os.path.splitext(pdb_path)[1].lower()
            parser = MMCIFParser(QUIET=True) if file_ext == '.cif' else PDBParser(QUIET=True)
            structure = parser.get_structure("mol", pdb_path)
            
            sequences = []
            nucleotide_map = {
                "A": "A", "DA": "A", "ADE": "A",
                "G": "G", "DG": "G", "GUA": "G",
                "C": "C", "DC": "C", "CYT": "C",
                "U": "U", "DU": "U", "URA": "U",
                "T": "T", "DT": "T", "THY": "T"
            }

            for model in structure:
                for chain in model:
                    chain_sequence = ''
                    for residue in chain.get_residues():
                        res_name = residue.get_resname().strip()
                        if res_name in nucleotide_map:
                            chain_sequence += nucleotide_map[res_name]
                        elif res_name in aa3to1:
                            chain_sequence += aa3to1[res_name]
                    
                    if chain_sequence:
                        sequences.append(chain_sequence)
            
            return max(sequences, key=len) if sequences else ""
    except Exception: 
        return ""

# --- Check Functions ---

def check_affinity_matching(config, mirna_data):
    print("\n--- 2. Checking Affinity Score Matching ---")
    raw_data_folder = os.path.join(config['project_root'], 'dataset', 'raw_data')
    affinity_folder = os.path.join(raw_data_folder, config['data_sources']['affinity']['folder'])
    affinity_cols = config['data_sources']['affinity']
    
    try:
        affinity_files = glob.glob(os.path.join(affinity_folder, "*.txt")) + glob.glob(os.path.join(affinity_folder, "*.csv"))
        if not affinity_files: raise FileNotFoundError("No affinity files found.")
        
        df_affinity = pd.concat([pd.read_csv(f, sep='\t' if f.lower().endswith(('.txt', '.tsv')) else ',') for f in affinity_files])
        affinity_ids = {str(val).strip().split()[0] for val in df_affinity[affinity_cols['id_col']]}
        matched_affinity = set(mirna_data.keys()).intersection(affinity_ids)
        
        print(f"  - ✅ Found {len(matched_affinity)} matching IDs between {len(mirna_data)} miRNAs and {len(affinity_ids)} affinity entries.")
        print(f"  - Match Rate: {len(matched_affinity)/len(mirna_data):.2%}")

    except Exception as e:
        print(f"  - ❌ ERROR during affinity check: {e}")

def check_conservation_matching(config, mirna_data):
    print("\n--- 3. Checking Conservation Score Matching ---")
    raw_data_folder = os.path.join(config['project_root'], 'dataset', 'raw_data')
    conservation_folder = os.path.join(raw_data_folder, config['data_sources']['conservation']['folder'])
    cons_cols = config['data_sources']['conservation']

    try:
        cons_files = glob.glob(os.path.join(conservation_folder, "*.txt"))
        if not cons_files: raise FileNotFoundError("No conservation files found.")

        df_cons = pd.concat([pd.read_csv(f, sep='\t') for f in cons_files])
        cons_families = {str(val).lower() for val in df_cons[cons_cols['id_col']]}
        
        mirna_families = set()
        for mid in mirna_data.keys():
            match = re.search(r"(mir-|let-)\d+[a-z]?", mid.lower())
            if match: mirna_families.add(match.group(0))
        
        matched_cons = mirna_families.intersection(cons_families)
        print(f"  - ✅ Found {len(matched_cons)} matching families between {len(mirna_families)} miRNA families and {len(cons_families)} conservation entries.")
        print(f"  - Match Rate: {len(matched_cons)/len(mirna_families):.2%}")

    except Exception as e:
        print(f"  - ❌ ERROR during conservation check: {e}")

def check_3d_structure_matching(config, target_data, competitor_data):
    print("\n--- 4. Checking 3D Structure (PDB) File Matching ---")
    pdb_base_dir = os.path.join(config['project_root'], 'dataset', 'pdb_files')
    
    for role, seq_data in [('target', target_data), ('competitor', competitor_data)]:
        pdb_dir = os.path.join(pdb_base_dir, f"{role}s")
        if not os.path.isdir(pdb_dir):
            print(f"  - ⚠️ INFO: PDB folder for '{role}s' not found, skipping check.")
            continue

        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb")) + glob.glob(os.path.join(pdb_dir, "*.cif"))
        if not pdb_files:
            print(f"  - ⚠️ INFO: No PDB/CIF files found in '{role}s' folder.")
            continue
            
        seq_data_values = set(seq_data.values())
        matched_by_id = 0
        matched_by_content = 0

        for pdb_path in pdb_files:
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
            
            if pdb_id in seq_data:
                matched_by_id += 1
                continue
            
            # --- FIX: Call the function with the correct name ---
            pdb_seq = get_sequence_from_pdb(pdb_path)
            if pdb_seq and pdb_seq.upper() in (s.upper() for s in seq_data_values):
                matched_by_content += 1
        
        total_matched = matched_by_id + matched_by_content
        print(f"For {role.capitalize()}s:")
        print(f"  - Total PDB/CIF files found: {len(pdb_files)}")
        print(f"  - Matched by ID: {matched_by_id}")
        print(f"  - Matched by Sequence Content: {matched_by_content}")
        print(f"  - ✅ Total Matched: {total_matched} / {len(pdb_files)} ({total_matched/len(pdb_files):.2%})")
        if total_matched < len(pdb_files):
            print(f"  - ⚠️ WARNING: {len(pdb_files) - total_matched} PDB files did not match any FASTA sequence.")

def check_config_logic(config):
    print("\n--- 5. Checking Config File Logic ---")
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

def main():
    """Runs a series of fast checks on the project setup."""
    print("--- Starting Final Project Setup Diagnostic Script ---\n")
    config = load_config()
    
    print("--- 1. Checking Paths and Loading Sequences ---")
    raw_data_folder = os.path.join(config['project_root'], 'dataset', 'raw_data')
    check_path(raw_data_folder, "'raw_data' subfolder")
    
    try:
        mirna_data = load_all_fasta_data(os.path.join(raw_data_folder, config['data_sources']['mirna']['folder']))
        target_data = load_all_fasta_data(os.path.join(raw_data_folder, config['data_sources']['rna_target']['folder']))
        competitor_data = load_all_fasta_data(os.path.join(raw_data_folder, config['data_sources']['protein_competitor']['folder']))
        print(f"  - ✅ Successfully loaded {len(mirna_data)} miRNA, {len(target_data)} Target, and {len(competitor_data)} Competitor sequences.")
    except Exception as e:
        print(f"  - ❌ FATAL ERROR loading sequence data: {e}")
        return

    check_affinity_matching(config, mirna_data)
    check_conservation_matching(config, mirna_data)
    check_3d_structure_matching(config, target_data, competitor_data)
    check_config_logic(config)

    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    main()