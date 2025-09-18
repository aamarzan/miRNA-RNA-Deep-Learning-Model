# s1a_prepare_dataset.py
# Final, unbiased, RAM-safe, multi-part Parquet, with chunked streaming and optional parallel row building

import os
import pandas as pd
from Bio import SeqIO
import numpy as np
import re
import json
from itertools import product, islice
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import random

# Reproducibility
random.seed(42)
np.random.seed(42)

# --- Import the processors library ---
# Must be available in PYTHONPATH or same directory as this script.
from molecule_processors import PROCESSOR_MAP

# ==============================
# CONFIGURATION LOADER
# ==============================
def load_config(config_path=None):
    if config_path is None:
        # Looks for config.json in the same directory as the script.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

# ==============================
# FILE AND DATA LOADERS
# ==============================
def _get_files_in_folder(folder_path, extensions):
    if not os.path.exists(folder_path):
        return []
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if any(f.lower().endswith(ext) for ext in extensions)
    ]

def load_data_from_fasta(folder_path):
    data_dict = {}
    file_paths = _get_files_in_folder(folder_path, ['.fasta', '.fa', '.fna', '.txt'])
    for filepath in file_paths:
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                data_dict[record.id] = str(record.seq)
        except Exception:
            # Silently skip malformed FASTA files, as in your original
            pass
    return data_dict

def load_scores(folder_path, id_col, score_col, file_type_name):
    data_dict = {}
    print(f"  Scanning {file_type_name} files in '{folder_path}'...")
    for filepath in _get_files_in_folder(folder_path, ['.txt', '.tsv', '.csv']):
        try:
            sep = '\t' if filepath.lower().endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(
                filepath, sep=sep, comment='#',
                usecols=[id_col, score_col], dtype={id_col: str},
                low_memory=False
            )
            df.dropna(inplace=True)
            for _, row in df.iterrows():
                data_dict[str(row[id_col])] = float(row[score_col])
        except Exception as e:
            print(f"    - Error loading score file {filepath}: {e}")
    return data_dict

# ==============================
# DIAGNOSTICS
# ==============================
def run_id_matching_diagnostics(mirna_data, affinity_data):
    """
    Cleans complex IDs and provides a detailed report on the match rate between
    miRNA sequences and affinity family/score priors.
    """
    print("\n--- Running ID Matching Diagnostics ---")
    if not mirna_data or not affinity_data:
        print("  - Skipping diagnostics: miRNA or affinity data not loaded.")
        return

    def normalize_id(key):
        return key.strip().split()[0]

    mirna_ids = {normalize_id(k) for k in mirna_data.keys()}
    affinity_ids = {normalize_id(k) for k in affinity_data.keys()}

    matched_ids = mirna_ids.intersection(affinity_ids)
    affinity_only_ids = affinity_ids - mirna_ids
    mirna_only_ids = mirna_ids - affinity_ids

    print(f"Total Unique miRNA IDs from FASTA (normalized): {len(mirna_ids)}")
    print(f"Total Unique IDs from Affinity File (normalized): {len(affinity_ids)}")
    print(f"  - Number of Matched IDs: {len(matched_ids)}")

    if affinity_only_ids:
        print(f"\nWARNING: {len(affinity_only_ids)} IDs are in the Affinity File but NOT in the miRNA FASTA file.")
        print(f"  (These scores will be ignored. Example IDs: {list(affinity_only_ids)[:5]})")

    if mirna_only_ids:
        print(f"\nNOTE: {len(mirna_only_ids)} miRNAs have sequences but NO affinity score.")
        print(f"  (They won’t get random labels anymore; pairwise labels will be computed.)")

    print("---------------------------------------")

# ==============================
# PROXY-LABEL HELPERS
# ==============================
def seed_match_score(mirna_u, target_u):
    m = (mirna_u or "").replace('T','U').upper()
    t = (target_u or "").replace('T','U').upper()
    if len(m) < 8 or len(t) < 8:
        return 0.0, 0
    seed = m[1:8]  # positions 2–8
    comp = str.maketrans('AUGC', 'UACG')
    seed_comp = seed.translate(comp)
    best = 0
    best_pos = 0
    for i in range(0, len(t) - 6):
        window = t[i:i+7]
        matches = sum(1 for a, b in zip(seed_comp, window) if a == b)
        if matches > best:
            best = matches
            best_pos = i
    return best / 7.0, best_pos

def gc_strength_score(mirna_u, target_u, start_pos):
    m = (mirna_u or "").replace('T','U').upper()
    t = (target_u or "").replace('T','U').upper()
    if len(m) < 8 or start_pos + 7 > len(t):
        return 0.0
    seed = m[1:8]
    comp = str.maketrans('AUGC', 'UACG')
    seed_comp = seed.translate(comp)
    window = t[start_pos:start_pos+7]
    score = 0.0
    for a, b in zip(seed_comp, window):
        if a == b:
            score += 1.0 if a in ('G', 'C') else 0.6
    return score / 7.0

def accessibility_score(dot_bracket_json, start_pos, length=7):
    try:
        vec = json.loads(dot_bracket_json or "[]")
    except Exception:
        vec = []
    if not vec or start_pos + length > len(vec):
        return 0.0
    # encoded: . -> 0 (unpaired), ( -> 1, ) -> -1
    unpaired = sum(1 for v in vec[start_pos:start_pos+length] if v == 0)
    return unpaired / float(length)

def normalize_feature_01(x, lo=0.0, hi=1.0):
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, v))

# ==============================
# COMPETITOR PROPENSITY (VECTORIZED)
# ==============================
_COMP_PROP_CACHE = {}
def competitor_propensity_score(comp_seq: str, target_seq: str) -> float:
    """
    Computes the max 7-mer matching score between competitor and target.
    Logic unchanged: max exact character matches over all 7-mer alignments, normalized by 7.
    Vectorized with NumPy for speed.
    """
    if not comp_seq or not target_seq or len(comp_seq) < 7 or len(target_seq) < 7:
        return 0.0

    key = (comp_seq, target_seq)
    if key in _COMP_PROP_CACHE:
        return _COMP_PROP_CACHE[key]

    comp = np.frombuffer(comp_seq.encode('ascii', 'ignore'), dtype=np.uint8)
    targ = np.frombuffer(target_seq.encode('ascii', 'ignore'), dtype=np.uint8)

    best = 0
    for i in range(len(comp) - 6):
        w = comp[i:i+7]
        for j in range(len(targ) - 6):
            matches = np.sum(w == targ[j:j+7])
            if matches > best:
                best = matches
                if best == 7:
                    break
        if best == 7:
            break

    score = float(best) / 7.0
    _COMP_PROP_CACHE[key] = score
    return score

# ==============================
# CHUNKED GENERATOR
# ==============================
def chunked(iterable, n):
    """
    Yield successive lists (chunks) of size n from iterable.
    """
    while True:
        chunk = list(islice(iterable, n))
        if not chunk:
            break
        yield chunk

# ==============================
# ROW BUILDER FOR PARALLELISM (pure function, picklable)
# ==============================
def build_rows_for_combo(args):
    """
    Build one or two rows for a single (primary, target, competitor) combo.
    Returns a list of rows (dicts).
    """
    primary_data, target_data, competitor_data, family_prior_map = args

    mirna_id = primary_data.get('id')
    mirna_seq = primary_data.get('sequence', '')
    target_seq = target_data.get('sequence', '')
    comp_seq = competitor_data.get('sequence', '')

    # Pairwise label components
    seed_score, seed_pos = seed_match_score(mirna_seq, target_seq)
    gc_score = gc_strength_score(mirna_seq, target_seq, seed_pos)
    acc_score = accessibility_score(target_data.get('structure_vector', '[]'), seed_pos, 7)

    fam_prior_raw = family_prior_map.get(mirna_id, 0.0)
    fam_prior = normalize_feature_01(fam_prior_raw, 0.0, 1.0)
    cons_raw = primary_data.get('conservation', 0.0)
    cons_n = normalize_feature_01(cons_raw, 0.0, 1.0)

    # Combine into pairwise label (weights unchanged)
    w_seed, w_gc, w_acc, w_prior, w_cons = 0.35, 0.15, 0.20, 0.20, 0.10
    L_pair = normalize_feature_01(
        w_seed * seed_score +
        w_gc * gc_score +
        w_acc * acc_score +
        w_prior * fam_prior +
        w_cons * cons_n,
        0.0, 1.0
    )

    # Competitor effect (weak supervision)
    comp_prop = competitor_propensity_score(comp_seq, target_seq) if comp_seq else 0.0
    beta = 0.6
    L_with_comp = max(0.0, L_pair - beta * comp_prop)

    # Baseline row (no competitor)
    row_base = {
        'primary_id': mirna_id,
        'primary_sequence': mirna_seq,
        'gc_content': primary_data.get('gc_content'),
        'dg': primary_data.get('dg'),
        'structure_vector': primary_data.get('structure_vector'),
        'adjacency_matrix': primary_data.get('adjacency_matrix'),
        'affinity': L_pair,
        'conservation': cons_n,
        'target_id': target_data.get('id'),
        'target_sequence': target_seq,
        'competitor_id': 'NO_COMPETITOR',
        'competitor_sequence': ''
    }

    rows = [row_base]
    if comp_seq:
        row_comp = dict(row_base)
        row_comp['competitor_id'] = competitor_data.get('id')
        row_comp['competitor_sequence'] = comp_seq
        row_comp['affinity'] = L_with_comp
        rows.append(row_comp)

    return rows

# ==============================
# MAIN DATASET PREPARATION
# ==============================
def prepare_dataset(config):
    start_time = time.time()

    PROJECT_ROOT = config['project_root']
    DATA_ROOT = os.path.join(PROJECT_ROOT, config['data_folders']['main_dataset_folder'])
    RAW_DATA_ROOT = os.path.join(DATA_ROOT, 'raw_data')
    PREPARED_DATASET_DIR = os.path.join(DATA_ROOT, config['data_folders']['prepared_subfolder'])
    os.makedirs(PREPARED_DATASET_DIR, exist_ok=True)
    PARAMS = {**config.get('processing_parameters', {}), **config.get('training_parameters', {})}

    print("--- Starting Universal Dataset Preparation ---")

    # --------------------------------------------
    # Step 1: Load data sources
    # --------------------------------------------
    print("\nStep 1: Loading all data sources from config...")
    data_sources = {}
    for name, source_info in config['data_sources'].items():
        if name.startswith('_'):
            continue
        path = os.path.join(RAW_DATA_ROOT, source_info['folder'])
        if source_info['type'] == 'fasta':
            data_sources[name] = load_data_from_fasta(path)
        elif source_info['type'] == 'score':
            data_sources[name] = load_scores(path, source_info['id_col'], source_info['score_col'], name.capitalize())
        else:
            print(f"  - Unknown data source type for '{name}': {source_info['type']}")

    run_id_matching_diagnostics(data_sources.get('mirna', {}), data_sources.get('affinity', {}))

    # --------------------------------------------
    # Step 2: Pre-processing all molecule types
    # --------------------------------------------
    print("\nStep 2: Pre-processing all molecule types...")
    processed_data = {}
    exp_setup = config['experiment_setup']

    # Sliding window parameters from config
    sw_params = PARAMS.get('sliding_window', {})
    use_sw = sw_params.get('use_sliding_window', False)
    window_size = sw_params.get('window_size', 500)
    step_size = sw_params.get('step_size', 250)

    for role, molecule_type in exp_setup.items():
        role_key = f"{molecule_type.lower()}_{role.replace('_molecule','')}" if 'molecule' in role else molecule_type.lower()
        if role == 'primary_molecule':
            role_key = molecule_type.lower()

        if role_key not in data_sources or not data_sources[role_key]:
            print(f"  - WARNING: No data found for '{role}' (source key: '{role_key}'). Skipping.")
            processed_data[role] = []
            continue

        print(f"  - Processing {role} ({molecule_type})...")
        raw_molecules = data_sources[role_key]

        # Apply sliding window to target if enabled
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
                processed_molecules.append((mol_id, seq))
        if use_sw:
            print()  # newline after slicing messages

        processor_func = PROCESSOR_MAP.get(molecule_type)
        if not processor_func:
            processed_data[role] = []
            continue

        # Prepare args as in your original structure
        with Pool(processes=cpu_count()) as pool:
            processing_args = [((mol_id, seq), PARAMS, role) for (mol_id, seq) in processed_molecules]
            results = list(tqdm(
                pool.imap(processor_func, processing_args),
                total=len(processing_args),
                desc=f"Processing {role}"
            ))

        processed_data[role] = [res for res in results if isinstance(res, dict)]
        print(f"    - {len(processed_data[role])} molecules/chunks passed processing.")

    primary_molecules = processed_data.get('primary_molecule', [])
    target_molecules = processed_data.get('target_molecule', [])
    competitor_molecules = processed_data.get('competitor_molecule', [])

    if not primary_molecules or not target_molecules:
        print("\nCRITICAL ERROR: No primary or target molecules remained after processing. Halting.")
        return

    # --------------------------------------------
    # Step 3: Separating Known vs. Unknown Primary Molecules
    # --------------------------------------------
    print("\nStep 3: Separating Known vs. Unknown Primary Molecules...")
    known_primary_molecules = []
    unknown_primary_molecules = []

    for molecule_data in primary_molecules:
        molecule_id = molecule_data['id']
        if data_sources.get('affinity', {}).get(molecule_id) is not None:
            known_primary_molecules.append(molecule_data)
        else:
            unknown_primary_molecules.append(molecule_data)

    print(f"  - Found {len(known_primary_molecules)} molecules with family prior scores.")
    print(f"  - Found {len(unknown_primary_molecules)} molecules with no prior (will be down-sampled).")

    # Down-sample unknowns to balance compute
    ratio = PARAMS.get('downsampling_ratio_unknown_to_known', 1.0)
    num_known = len(known_primary_molecules)
    num_unknown_to_keep = min(len(unknown_primary_molecules), int(max(1, num_known) * ratio))

    print(f"\nDown-sampling unknowns from {len(unknown_primary_molecules)} to {num_unknown_to_keep} (Known:Unknown Ratio ≈ 1:{ratio})...")
    unknown_subsample = random.sample(unknown_primary_molecules, num_unknown_to_keep) if num_unknown_to_keep > 0 else []

    balanced_primary_molecules = known_primary_molecules + unknown_subsample
    random.shuffle(balanced_primary_molecules)
    print(f"  - Created a balanced primary molecule set of {len(balanced_primary_molecules)} total entries.")

    # --------------------------------------------
    # Step 4: Augmenting datasets with conservation (no random labels)
    # --------------------------------------------
    print("\nStep 4: Augmenting datasets with conservation (no random labels)")
    # Only set conservation on primary molecules; do NOT set 'affinity' here anymore
    for molecule_data in balanced_primary_molecules:
        molecule_id = molecule_data['id']
        # Conservative family mapping: try to extract family-like key
        mirna_family_match = re.search(r"hsa-mir-\d+[a-z]?", molecule_id.lower())
        mirna_family_name = mirna_family_match.group(0) if mirna_family_match else molecule_id.lower()
        molecule_data['conservation'] = data_sources.get('conservation', {}).get(mirna_family_name, 0.0)
    print("  - Augmentation complete.")

    # --------------------------------------------
    # Step 5: Preparing Final Competitor List
    # --------------------------------------------
    print("\nStep 5: Preparing Final Competitor List...")
    null_competitor = {
        'id': 'NO_COMPETITOR', 'sequence': '', 'original_sequence': '',
        'gc_content': 0.0, 'dg': 0.0, 'structure_vector': '[]', 'adjacency_matrix': '[]'
    }
    competitors_augmented = competitor_molecules + [null_competitor]
    print(f"  - Final competitor list contains {len(competitors_augmented)} entries (including null).")

    # =========================================================================================
    # STEP 6: GENERATING AND SHUFFLING COMBINATIONS (RAM-safe, chunked)
    # =========================================================================================
    print("\nStep 6: Generating and Shuffling Combinations with Progress...")
    total_combinations_count = len(balanced_primary_molecules) * len(target_molecules) * len(competitors_augmented)
    print(f"  - Preparing to generate {total_combinations_count:,} combinations...")

    from itertools import islice

    def chunked(iterable, n):
        while True:
            chunk = list(islice(iterable, n))
            if not chunk:
                break
            yield chunk

    dp = config.get('data_processing', {})
    PARQUET_BATCH_SIZE = int(dp.get('batch_size_parquet', 2_000_000))  # bigger batches = fewer flushes
    CHUNK_SIZE = max(20_000, PARQUET_BATCH_SIZE)  # shuffle in big blocks
    PARQUET_COMPRESSION = dp.get('parquet_compression', 'snappy')
    PARQUET_PREFIX = dp.get('parquet_output_prefix', 'training_combinations')

    output_folder = PREPARED_DATASET_DIR
    os.makedirs(output_folder, exist_ok=True)
    output_base = os.path.join(output_folder, PARQUET_PREFIX)

    buffer = []
    part_idx = 0
    total_rows_written = 0
    parquet_schema = None

    def flush_parquet_buffer():
        global buffer, part_idx, total_rows_written, parquet_schema
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=parquet_schema)
        if parquet_schema is None:
            parquet_schema = table.schema
            table = table.cast(parquet_schema)
        else:
            table = table.cast(parquet_schema)
        part_path = f"{output_base}_part{part_idx:05d}.parquet"
        pq.write_table(table, part_path, compression=PARQUET_COMPRESSION)
        total_rows_written += len(buffer)
        buffer.clear()
        part_idx += 1

    # =========================================================================================
    # STEP 7: STREAMING SHUFFLED COMBINATIONS TO PARQUET (pairwise labels + competitor effect)
    # =========================================================================================
    print("\nStep 7: Streaming Shuffled Combinations to Parquet with Progress...")
    start_write_time = time.time()

    # --- Diagnostics before writing ---
    # Estimate competitor fraction from a small sample
    sample_size = min(10000, len(all_combinations))
    sample_with_comp = sum(1 for p, t, c in all_combinations[:sample_size] if c.get('sequence', ''))
    competitor_fraction = sample_with_comp / sample_size if sample_size else 0.0

    est_total_rows = int(total_combinations_count * (1 + competitor_fraction))
    est_parts = math.ceil(est_total_rows / PARQUET_BATCH_SIZE)

    print(f"  - Competitor fraction (sampled): {competitor_fraction:.2%}")
    print(f"  - Estimated total rows: {est_total_rows:,}")
    print(f"  - PARQUET_BATCH_SIZE: {PARQUET_BATCH_SIZE:,}")
    print(f"  - Estimated number of part files: {est_parts:,}")

    # --- Output setup ---
    output_base = os.path.join(PREPARED_DATASET_DIR, PARQUET_PREFIX)
    buffer = []
    part_idx = 0
    total_rows_written = 0
    parquet_schema = None  # lock schema after first batch

    def flush_parquet_buffer():
        nonlocal buffer, part_idx, total_rows_written, parquet_schema
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=parquet_schema)
        if parquet_schema is None:
            parquet_schema = table.schema
            table = table.cast(parquet_schema)
        else:
            table = table.cast(parquet_schema)
        part_path = f"{output_base}_part{part_idx:05d}.parquet"
        pq.write_table(table, part_path, compression=PARQUET_COMPRESSION)
        total_rows_written += len(buffer)
        buffer.clear()
        part_idx += 1

    # --- Family prior map ---
    family_prior_map = data_sources.get('affinity', {})

    pbar = tqdm(total=total_combinations_count, desc="  Writing to Parquet")

    for primary_data, target_data, competitor_data in all_combinations:
        mirna_id = primary_data.get('id')
        mirna_seq = primary_data.get('sequence', '')
        target_seq = target_data.get('sequence', '')
        comp_seq = competitor_data.get('sequence', '')

        # Pairwise label components
        seed_score, seed_pos = seed_match_score(mirna_seq, target_seq)
        gc_score = gc_strength_score(mirna_seq, target_seq, seed_pos)
        acc_score = accessibility_score(target_data.get('structure_vector','[]'), seed_pos, 7)

        fam_prior_raw = family_prior_map.get(mirna_id, 0.0)
        fam_prior = normalize_01(fam_prior_raw, 0.0, 1.0)
        cons_raw = primary_data.get('conservation', 0.0)
        cons_n = normalize_01(cons_raw, 0.0, 1.0)

        # Combine into pairwise label
        w_seed, w_gc, w_acc, w_prior, w_cons = 0.35, 0.15, 0.20, 0.20, 0.10
        L_pair = normalize_01(
            w_seed*seed_score + w_gc*gc_score + w_acc*acc_score + w_prior*fam_prior + w_cons*cons_n,
            0.0, 1.0
        )

        # Competitor effect
        comp_prop = competitor_propensity_score(comp_seq, target_seq) if comp_seq else 0.0
        beta = 0.6
        L_with_comp = max(0.0, L_pair - beta * comp_prop)

        # Baseline row
        row_base = {
            'primary_id': mirna_id, 'primary_sequence': mirna_seq,
            'gc_content': primary_data.get('gc_content'), 'dg': primary_data.get('dg'),
            'structure_vector': primary_data.get('structure_vector'), 'adjacency_matrix': primary_data.get('adjacency_matrix'),
            'affinity': L_pair, 'conservation': cons_n,
            'target_id': target_data.get('id'), 'target_sequence': target_seq,
            'competitor_id': 'NO_COMPETITOR', 'competitor_sequence': ''
        }
        buffer.append(row_base)

        # With competitor
        if comp_seq:
            row_comp = dict(row_base)
            row_comp['competitor_id'] = competitor_data.get('id')
            row_comp['competitor_sequence'] = comp_seq
            row_comp['affinity'] = L_with_comp
            buffer.append(row_comp)

        # Flush if batch full
        if len(buffer) >= PARQUET_BATCH_SIZE:
            flush_parquet_buffer()

        pbar.update(1)

    # Final flush
    flush_parquet_buffer()
    pbar.close()

    end_write_time = time.time()
    print(f"\n  - Wrote {total_rows_written:,} rows across {part_idx} Parquet part files (prefix: {os.path.basename(output_base)}).")
    print(f"  - Time taken for this step: {end_write_time - start_write_time:.2f} seconds")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    config = load_config()
    prepare_dataset(config)
