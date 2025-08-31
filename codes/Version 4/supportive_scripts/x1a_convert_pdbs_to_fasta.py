# convert_pdbs_to_fasta.py
# PURPOSE:
# To automate the creation of a matched FASTA sequence file directly from a
# directory of curated PDB/mmCIF structure files.

import os
import glob
import warnings
from Bio.PDB import PDBParser, MMCIFParser, PDBExceptions
from Bio.PDB.Polypeptide import protein_letters_3to1_extended as aa3to1

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the directory containing your PDB/CIF files.
INPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/pdb_files/targets"

# 2. Define the name for the output FASTA file.
OUTPUT_FASTA_FILE = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/target/select/targets_from_pdb.fasta"
# --- END OF CONFIGURATION ---

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
                "A": "A", "DA": "A", "ADE": "A", "G": "G", "DG": "G", "GUA": "G",
                "C": "C", "DC": "C", "CYT": "C", "U": "U", "DU": "U", "URA": "U",
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

def create_fasta_from_pdbs():
    """
    Main function to find PDBs, extract sequences, and write a FASTA file.
    """
    print(f"--- Creating FASTA file from PDBs in: {INPUT_DIR} ---")
    
    pdb_files = glob.glob(os.path.join(INPUT_DIR, "*.pdb")) + glob.glob(os.path.join(INPUT_DIR, "*.cif"))

    if not pdb_files:
        print(f"❌ ERROR: No .pdb or .cif files found in '{INPUT_DIR}'")
        return
        
    processed_count = 0
    with open(OUTPUT_FASTA_FILE, 'w') as f_out:
        for pdb_path in pdb_files:
            filename = os.path.basename(pdb_path)
            pdb_id = os.path.splitext(filename)[0]
            
            print(f"  - Processing {filename}...")
            sequence = get_sequence_from_pdb(pdb_path)
            
            if sequence:
                f_out.write(f">{pdb_id}\n")
                # Write sequence in lines of 60 characters for standard FASTA format
                for i in range(0, len(sequence), 60):
                    f_out.write(sequence[i:i+60] + "\n")
                processed_count += 1
            else:
                print(f"    - ⚠️ WARNING: Could not extract a valid sequence from {filename}.")
                
    print(f"\n✅ Success! Processed {processed_count} PDB files.")
    print(f"   FASTA file saved to: {OUTPUT_FASTA_FILE}")

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FASTA_FILE), exist_ok=True)
    create_fasta_from_pdbs()