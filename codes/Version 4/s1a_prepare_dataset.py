# s1_prepare_dataset.py (Definitive, Final Corrected Version)
import os
import pandas as pd
from Bio import SeqIO
import numpy as np
import re
import json
from itertools import product
from multiprocessing import Pool, cpu_count
import time
import pyarrow as pa
import pyarrow.parquet as pq

# --- Import the new processors library ---
from molecule_processors import PROCESSOR_MAP

# --- Configuration Loader ---
def load_config(config_path=None):
    if config_path is None:
        # Looks for config.json in the same directory as the script.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

# --- Helper Functions ---
def _get_files_in_folder(folder_path, extensions):
    if not os.path.exists(folder_path): return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in extensions)]

def load_data_from_fasta(folder_path):
    data_dict = {}
    file_paths = _get_files_in_folder(folder_path, ['.fasta', '.fa', '.fna', '.txt'])
    for filepath in file_paths:
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                data_dict[record.id] = str(record.seq)
        except Exception: pass
    return data_dict

def load_scores(folder_path, id_col, score_col, file_type_name):
    data_dict = {}
    print(f"  Scanning {file_type_name} files in '{folder_path}'...")
    for filepath in _get_files_in_folder(folder_path, ['.txt', '.tsv', '.csv']):
        try:
            sep = '\t' if filepath.lower().endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(filepath, sep=sep, comment='#', usecols=[id_col, score_col], dtype={id_col: str}, low_memory=False)
            df.dropna(inplace=True)
            for _, row in df.iterrows():
                data_dict[str(row[id_col])] = float(row[score_col])
        except Exception as e:
            print(f"    - Error loading score file {filepath}: {e}")
    return data_dict

# --- Main Dataset Preparation Function ---
def prepare_dataset(config):
    start_time = time.time()
    
    PROJECT_ROOT = config['project_root']
    DATA_ROOT = os.path.join(PROJECT_ROOT, config['data_folders']['main_dataset_folder'])
    PREPARED_DATASET_DIR = os.path.join(DATA_ROOT, config['data_folders']['prepared_subfolder'])
    os.makedirs(PREPARED_DATASET_DIR, exist_ok=True)
    PARAMS = {**config.get('processing_parameters', {}), **config.get('training_parameters', {})}

    print("--- Starting Universal Dataset Preparation ---")
    
    print("\nStep 1: Loading all data sources from config...")
    data_sources = {}
    for name, source_info in config['data_sources'].items():
        if name.startswith('_'): continue
        path = os.path.join(DATA_ROOT, 'raw_data', source_info['folder'])
        if source_info['type'] == 'fasta':
            data_sources[name] = load_data_from_fasta(path)
        elif source_info['type'] == 'score':
            data_sources[name] = load_scores(path, source_info['id_col'], source_info['score_col'], name.capitalize())
    
    print("\nStep 2: Pre-processing all molecule types...")
    processed_data = {}
    exp_setup = config['experiment_setup']
    
    # --- NEW: Get window parameters from config ---
    sw_params = PARAMS.get('sliding_window', {})
    use_sw = sw_params.get('use_sliding_window', False)
    window_size = sw_params.get('window_size', 500)
    step_size = sw_params.get('step_size', 250)

    for role, molecule_type in exp_setup.items():
        role_key = f"{molecule_type.lower()}_{role.replace('_molecule','')}" if 'molecule' in role else molecule_type.lower()
        if role == 'primary_molecule': role_key = molecule_type.lower()

        if role_key not in data_sources or not data_sources[role_key]:
            print(f"  - WARNING: No data found for '{role}' (source key: '{role_key}'). Skipping.")
            processed_data[role] = []
            continue

        print(f"  - Processing {role} ({molecule_type})...")
        raw_molecules = data_sources[role_key]
        
        # --- NEW: Apply sliding window logic to generate chunks ---
        processed_molecules = []
        for mol_id, seq in raw_molecules.items():
            if use_sw and role == 'target_molecule' and len(seq) > window_size:
                num_chunks = 0
                for i in range(0, len(seq) - window_size + 1, step_size):
                    chunk_seq = seq[i:i + window_size]
                    chunk_id = f"{mol_id}_chunk_{i}"
                    processed_molecules.append((chunk_id, chunk_seq))
                    num_chunks += 1
                print(f"    - Sliced target {mol_id} (len {len(seq)}) into {num_chunks} chunks of size {window_size}", end='\r')
            else:
                # If not using sliding window or sequence is too short, use it as is
                processed_molecules.append((mol_id, seq))
        if use_sw: print() # Newline after slicing messages
        
        processor_func = PROCESSOR_MAP.get(molecule_type)
        if not processor_func: continue

        with Pool(processes=cpu_count()) as pool:
            # Pass the tuple (id, seq) to the processor
            results = pool.map(processor_func, [(mol_data, PARAMS, role) for mol_data in processed_molecules])
        
        processed_data[role] = [res for res in results if isinstance(res, dict)]
        print(f"    - {len(processed_data[role])} molecules/chunks passed processing.")

    primary_molecules = processed_data.get('primary_molecule', [])
    target_molecules = processed_data.get('target_molecule', [])
    competitor_molecules = processed_data.get('competitor_molecule', [])

    if not primary_molecules or not target_molecules:
        print("\nCRITICAL ERROR: No primary or target molecules remained after processing. Halting.")
        return

    print("\nStep 3: Augmenting primary molecules with scores...")
    for molecule_data in primary_molecules:
        molecule_id = molecule_data['id']
        molecule_data['affinity'] = data_sources.get('affinity', {}).get(molecule_id, 0.0)
        mirna_family_match = re.search(r"mir-\d+[a-z]?", molecule_id.lower())
        mirna_family = mirna_family_match.group(0) if mirna_family_match else molecule_id.lower()
        molecule_data['conservation'] = data_sources.get('conservation', {}).get(mirna_family, 0.0)
    print("  - Augmentation complete.")

    print("\nStep 4: Generating and streaming combinations to Parquet...")
    output_filename = f"Prepared_Dataset_{int(time.time())}.parquet"
    output_path = os.path.join(PREPARED_DATASET_DIR, output_filename)
    
    null_competitor = {'id': 'NO_COMPETITOR', 'sequence': '', 'original_sequence': '', 'gc_content': 0.0, 'dg': 0.0, 'structure_vector': '[]', 'adjacency_matrix': '[]'}
    competitors_augmented = competitor_molecules + [null_competitor]
    
    parquet_writer = None
    batch, total_rows = [], 0
    
    combinations = product(primary_molecules, target_molecules, competitors_augmented)

    for primary_data, target_data, competitor_data in combinations:
        row = {
            'primary_id': primary_data.get('id'), 'primary_sequence': primary_data.get('sequence'),
            'gc_content': primary_data.get('gc_content'), 'dg': primary_data.get('dg'),
            'structure_vector': primary_data.get('structure_vector'), 'adjacency_matrix': primary_data.get('adjacency_matrix'),
            'affinity': primary_data.get('affinity'), 'conservation': primary_data.get('conservation'),
            'target_id': target_data.get('id'), 'target_sequence': target_data.get('sequence'),
            'competitor_id': competitor_data.get('id'), 'competitor_sequence': competitor_data.get('sequence')
        }
        batch.append(row)

        if len(batch) >= PARAMS.get('batch_size_parquet', 50000):
            table = pa.Table.from_pandas(pd.DataFrame(batch), preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_path, table.schema)
            parquet_writer.write_table(table)
            total_rows += len(batch)
            print(f"  ... {total_rows} rows written", end='\r')
            batch = []

    if batch:
        table = pa.Table.from_pandas(pd.DataFrame(batch), preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_path, table.schema)
        parquet_writer.write_table(table)
        total_rows += len(batch)

    if parquet_writer: parquet_writer.close()
    
    end_time = time.time()
    print(f"\n\n--- Dataset Preparation Summary ---")
    print(f"Total combinations generated: {total_rows}")
    print(f"Dataset saved to {output_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    config = load_config()
    prepare_dataset(config)