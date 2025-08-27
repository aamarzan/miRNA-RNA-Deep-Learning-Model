# codes/processors.py (With Intelligent PDB Matching)
import os
import re
import json
import numpy as np
import subprocess
import random
import io
import warnings
from Bio.PDB import PDBParser, MMCIFParser, PDBExceptions
from Bio.PDB.Polypeptide import seq1

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

# --- NEW: Helper to extract sequence from a PDB/mmCIF file ---
def _get_sequence_from_pdb(pdb_path):
    """Extracts the canonical 1-letter sequence from a PDB or mmCIF file."""
    try:
        # Suppress noisy PDB warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBExceptions.PDBConstructionWarning)
            
            file_ext = os.path.splitext(pdb_path)[1].lower()
            parser = MMCIFParser() if file_ext == '.cif' else PDBParser()
            structure = parser.get_structure("mol", pdb_path)
            
            sequences = []
            for model in structure:
                for chain in model:
                    # Use seq1 to convert 3-letter codes to 1-letter, handling heteroatoms
                    residues = [res for res in chain if res.id[0] == ' '] # Standard residues
                    sequences.append(seq1("".join([res.get_resname() for res in residues]), custom_map={"HOH":""}))
            
            # Return the longest sequence found in the structure
            return max(sequences, key=len) if sequences else ""
    except Exception:
        return ""

# --- Feature Calculation Functions ---
def calculate_gc_content(sequence):
    """Calculates the GC content of a sequence."""
    if not sequence: return 0.0
    return (sequence.upper().count('G') + sequence.upper().count('C')) / len(sequence)

def predict_rna_structure_1d(sequence):
    """Calculates 1D structure vector and dG for an RNA sequence using RNAfold."""
    config = load_config()
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
    except subprocess.CalledProcessError as e:
        # --- ENHANCED LOGGING ---
        print(f"  - WARNING: RNAfold failed for sequence. STDERR: {e.stderr}")
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
    for i in range(seq_len - 1):
        adjacency_matrix[i, i + 1] = 1
        adjacency_matrix[i + 1, i] = 1
    return adjacency_matrix

# --- UPDATED: predict_graph_structure with intelligent matching ---
def predict_graph_structure(molecule_id, sequence, role):
    """Generates a graph (adjacency matrix) for a molecule."""
    config = load_config()
    if not config.get('processing_parameters', {}).get('enable_pdb_processing', False):
        return None 
        
    pdb_path = None
    role_key = role.replace('_molecule', '')
    pdb_folder_path = os.path.join(config.get('project_root', '.'), config.get('structure_folders', {}).get(role_key, ''))

    if os.path.isdir(pdb_folder_path):
        for ext in ['.pdb', '.cif']:
            potential_path = os.path.join(pdb_folder_path, f"{molecule_id}{ext}")
            if os.path.exists(potential_path):
                pdb_path = potential_path
                break
        
        if not pdb_path:
            for filename in os.listdir(pdb_folder_path):
                if filename.lower().endswith(('.pdb', '.cif')):
                    full_path = os.path.join(pdb_folder_path, filename)
                    pdb_seq = _get_sequence_from_pdb(full_path).upper()
                    if pdb_seq and pdb_seq == sequence.upper().replace('U','T'):
                        pdb_path = full_path
                        break 
    
    if pdb_path:
        dssr_cmd = config.get('tool_paths', {}).get('dssr') or 'x3dna-dssr'
        try:
            result = subprocess.run([dssr_cmd, f'--input={pdb_path}'], capture_output=True, text=True, check=True, timeout=60)
            match = re.search(r'secondary structure in dot-bracket notation\s*\n\s*(\S+)', result.stdout)
            if match:
                return _parse_dot_bracket_to_adjacency(match.group(1))
        except subprocess.CalledProcessError as e:
             # --- ENHANCED LOGGING ---
            print(f"  - WARNING: DSSR failed for {molecule_id} ({pdb_path}). STDERR: {e.stderr}")
            pass

    try:
        rnafold_cmd = config.get('tool_paths', {}).get('rnafold') or 'RNAfold'
        result = subprocess.run([rnafold_cmd], input=sequence, text=True, capture_output=True, check=True, encoding='utf-8', timeout=30)
        if len(result.stdout.strip().split('\n')) >= 2:
            return _parse_dot_bracket_to_adjacency(result.stdout.strip().split('\n')[1].split(' ')[0])
    except subprocess.CalledProcessError as e:
        # --- ENHANCED LOGGING ---
        print(f"  - WARNING: RNAfold (fallback) failed for {molecule_id}. STDERR: {e.stderr}")
    return None

def load_codon_table(table_path):
    """Loads a codon usage table from a text file into a dictionary."""
    codon_map = {}
    try:
        with open(table_path, 'r') as f:
            for line in f:
                parts = re.findall(r'([A-Z]{3})\s+([A-Z\*])\s+([\d\.]+)', line)
                for codon, aa, freq in parts:
                    if aa not in codon_map:
                        codon_map[aa] = []
                    codon_map[aa].append({'codon': codon.replace('T', 'U'), 'freq': float(freq)})
    except FileNotFoundError:
        print(f"  - WARNING: Codon usage table not found at {table_path}. Reverse translation will fail.")
        return None
    for aa, codons in codon_map.items():
        total_freq = sum(c['freq'] for c in codons)
        if total_freq > 0:
            for c in codons: c['prob'] = c['freq'] / total_freq
        else:
            for c in codons: c['prob'] = 1.0 / len(codons)
    return codon_map

def reverse_translate(aa_sequence, codon_map):
    """Converts an amino acid sequence to a probable nucleotide sequence."""
    if not codon_map: return ""
    nt_sequence = []
    for aa in aa_sequence.upper():
        if aa in codon_map:
            codons = [c['codon'] for c in codon_map[aa]]
            probabilities = [c['prob'] for c in codon_map[aa]]
            chosen_codon = random.choices(codons, weights=probabilities, k=1)[0]
            nt_sequence.append(chosen_codon)
    return "".join(nt_sequence)

def detect_sequence_type(sequence):
    """Detects if a sequence is RNA or Protein based on its alphabet."""
    rna_alphabet = set("ACGTUN")
    protein_alphabet = set("LIVFWYMCAGPSTHRKQNDE")
    seq_set = set(sequence.upper())
    if not seq_set.issubset(rna_alphabet) and seq_set.intersection(protein_alphabet):
        return "protein"
    return "rna"

# --- Main Universal Processor ---
def process_molecule_universal(args):
    """Unified processor for all molecule types."""
    (molecule_id, sequence), params, role = args
    config = load_config()
    
    if detect_sequence_type(sequence) == "protein":
        # --- CONFIGURABLE PATH ---
        codon_table_path = os.path.join(config.get('project_root', '.'), config.get('file_paths', {}).get('codon_table'))
        codon_map = load_codon_table(codon_table_path)
        nt_sequence = reverse_translate(sequence, codon_map)
        if not nt_sequence: return (molecule_id, "reject_reverse_translation")
    else:
        nt_sequence = sequence.replace('T', 'U')

    structural_features_1d = predict_rna_structure_1d(nt_sequence)
    if structural_features_1d is None:
        return (molecule_id, "reject_structure_1d")

    adjacency_matrix = predict_graph_structure(molecule_id, nt_sequence, role)
    if adjacency_matrix is None:
        adjacency_matrix = np.zeros((len(nt_sequence), len(nt_sequence)), dtype=int)
        
    return {
        'id': molecule_id, 'original_sequence': sequence, 'sequence': nt_sequence,
        'gc_content': calculate_gc_content(nt_sequence), **structural_features_1d,
        'adjacency_matrix': json.dumps(adjacency_matrix.tolist())
    }

PROCESSOR_MAP = {
    "miRNA": process_molecule_universal,
    "RNA": process_molecule_universal,
    "protein": process_molecule_universal,
}