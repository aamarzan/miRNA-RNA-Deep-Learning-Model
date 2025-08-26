# codes/processors.py (With Development Switch for DSSR)
import os
import re
import json
import numpy as np
import subprocess
import random

# --- Configuration Loader ---
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# --- Feature Calculation Functions ---
def calculate_gc_content(sequence):
    """Calculates the GC content of a sequence."""
    if not sequence: return 0.0
    return (sequence.upper().count('G') + sequence.upper().count('C')) / len(sequence)

def predict_rna_structure_1d(sequence):
    """Calculates 1D structure vector and dG for an RNA sequence using RNAfold."""
    config = load_config()
    # Correctly uses the configured tool path, falling back to 'RNAfold' if not specified
    rnafold_cmd = config.get('tool_paths', {}).get('rnafold') or 'RNAfold'
    try:
        result = subprocess.run([rnafold_cmd], input=sequence, text=True, capture_output=True, check=True, encoding='utf-8', timeout=60)
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            struct_line = output_lines[1]
            structure = struct_line.split(' ')[0]
            match = re.search(r"[-+]?\d+\.\d+", struct_line)
            dg = float(match.group(0)) if match else 0.0
            encoded_structure = [({'.': 0, '(': 1, ')': -1}).get(c, 0) for c in structure]
            return {'structure_vector': json.dumps(encoded_structure), 'dg': dg}
    except Exception as e:
        print(f"  - WARNING: RNAfold failed for sequence. Error: {e}")
        pass
    return None

def _parse_dot_bracket_to_adjacency(dbn_structure):
    """Converts a dot-bracket string to a binary adjacency matrix for GNNs."""
    seq_len = len(dbn_structure)
    adjacency_matrix = np.zeros((seq_len, seq_len), dtype=int)
    stack = []
    for i, char in enumerate(dbn_structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    # Add connections for the sequence backbone
    for i in range(seq_len - 1):
        adjacency_matrix[i, i + 1] = 1
        adjacency_matrix[i + 1, i] = 1
    return adjacency_matrix

def predict_graph_structure(molecule_id, sequence):
    """
    Generates a graph (adjacency matrix) for a molecule.
    1. If enabled, checks config for a PDB file and processes it with DSSR.
    2. Falls back to RNAfold prediction if PDB processing is disabled or fails.
    """
    config = load_config()
    # Correctly uses the configured tool path, falling back to 'x3dna-dssr' if not specified
    dssr_cmd = config.get('tool_paths', {}).get('dssr') or 'x3dna-dssr'
    use_pdb = config.get('processing_parameters', {}).get('enable_pdb_processing', False)

    if use_pdb and 'structure_files' in config and molecule_id in config['structure_files']:
        pdb_path = config['structure_files'][molecule_id]
        if os.path.exists(pdb_path):
            try:
                result = subprocess.run([dssr_cmd, f'--input={pdb_path}'],
                                        capture_output=True, text=True, check=True, timeout=60)
                match = re.search(r'secondary structure in dot-bracket notation\s*\n\s*(\S+)', result.stdout)
                if match:
                    dot_bracket_string = match.group(1)
                    return _parse_dot_bracket_to_adjacency(dot_bracket_string)
            except Exception as e:
                print(f"  - WARNING: DSSR failed for {molecule_id}. Falling back to RNAfold. Error: {e}")
                pass 

    # Fallback to RNAfold prediction
    try:
        rnafold_cmd = config.get('tool_paths', {}).get('rnafold') or 'RNAfold'
        result = subprocess.run([rnafold_cmd], input=sequence, text=True, capture_output=True, check=True, encoding='utf-8', timeout=30)
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            structure = output_lines[1].split(' ')[0]
            return _parse_dot_bracket_to_adjacency(structure)
    except Exception:
        pass
        
    return None

# <<< NEW: Function to load and parse a codon usage table >>>
def load_codon_table(table_path):
    """
    Loads a codon usage table from a text file into a dictionary.
    Format expected: UUU F 17.6 (1059)
    """
    codon_map = {}
    try:
        with open(table_path, 'r') as f:
            for line in f:
                parts = re.findall(r'([A-Z]{3})\s+([A-Z*])\s+([\d\.]+)', line)
                if parts:
                    codon, aa, freq = parts[0]
                    if aa not in codon_map:
                        codon_map[aa] = []
                    codon_map[aa].append({'codon': codon.replace('T', 'U'), 'freq': float(freq)})
    except FileNotFoundError:
        print(f"  - WARNING: Codon usage table not found at {table_path}. Reverse translation will fail.")
        return None
    
    # Normalize frequencies for probabilistic selection
    for aa, codons in codon_map.items():
        total_freq = sum(c['freq'] for c in codons)
        for c in codons:
            c['prob'] = c['freq'] / total_freq
            
    return codon_map

# <<< NEW: Core function to perform reverse translation >>>
def reverse_translate(aa_sequence, codon_map):
    """
    Converts an amino acid sequence to a probable nucleotide sequence
    based on codon usage frequencies.
    """
    if not codon_map:
        return ""
        
    nt_sequence = []
    for aa in aa_sequence.upper():
        if aa in codon_map:
            codons = [c['codon'] for c in codon_map[aa]]
            probabilities = [c['prob'] for c in codon_map[aa]]
            # Choose a codon based on its usage probability
            chosen_codon = random.choices(codons, weights=probabilities, k=1)[0]
            nt_sequence.append(chosen_codon)
    
    return "".join(nt_sequence)

# <<< NEW: Helper function to automatically detect sequence type >>>
def detect_sequence_type(sequence):
    """Detects if a sequence is RNA or Protein based on its alphabet."""
    rna_alphabet = set("ACGTUN")
    protein_alphabet = set("LIVFWYMCAGPSTHRKQNDE")
    
    seq_set = set(sequence.upper())
    
    # If any character is exclusively in the protein alphabet, it's a protein
    if not seq_set.issubset(rna_alphabet) and seq_set.intersection(protein_alphabet):
        return "protein"
    return "rna"

# <<< NEW: Single, unified processor for all molecule types >>>
def process_molecule_universal(args):
    """
    Unified processor that auto-detects sequence type (RNA or Protein).
    If Protein, it reverse-translates. Then, it processes the resulting
    nucleotide sequence to generate all features (GC, 1D/2D structure).
    """
    (molecule_id, sequence), params, role = args
    config = load_config()

    # --- Step 1: Auto-detect and handle sequence type ---
    seq_type = detect_sequence_type(sequence)
    
    if seq_type == "protein":
        codon_table_path = os.path.join(config.get('project_root', '.'), 'dataset', 'codon_tables', 'human_codon_usage.txt')
        codon_map = load_codon_table(codon_table_path)
        nt_sequence = reverse_translate(sequence, codon_map)
        if not nt_sequence:
            return (molecule_id, "reject_reverse_translation")
    else: # It's an RNA sequence
        nt_sequence = sequence.replace('T', 'U')

    # --- Step 2: Process the nucleotide sequence to generate all features ---
    gc = calculate_gc_content(nt_sequence)
    structural_features_1d = predict_rna_structure_1d(nt_sequence)
    if structural_features_1d is None:
        return (molecule_id, "reject_structure_1d")

    adjacency_matrix = predict_graph_structure(molecule_id, nt_sequence)
    if adjacency_matrix is None:
        adjacency_matrix = np.zeros((len(nt_sequence), len(nt_sequence)), dtype=int)
        
    serialized_adjacency = json.dumps(adjacency_matrix.tolist())

    return {
        'id': molecule_id,
        'original_sequence': sequence,
        'sequence': nt_sequence,
        'gc_content': gc,
        **structural_features_1d,
        'adjacency_matrix': serialized_adjacency
    }

    # --- GNN Feature Generation ---
    adjacency_matrix = predict_graph_structure(molecule_id, sequence)
    
    if adjacency_matrix is None:
        seq_len = len(sequence)
        adjacency_matrix = np.zeros((seq_len, seq_len), dtype=int)
        
    serialized_adjacency = json.dumps(adjacency_matrix.tolist())

    return {
        'id': molecule_id,
        'sequence': sequence,
        'gc_content': gc,
        **structural_features_1d,
        'adjacency_matrix': serialized_adjacency
    }

PROCESSOR_MAP = {
    "miRNA": process_molecule_universal,
    "RNA": process_molecule_universal,
    "protein": process_molecule_universal,
}