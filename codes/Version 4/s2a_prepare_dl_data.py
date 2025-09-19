#!/usr/bin/env python3
# s2a_prepare_dl_data.py
# Deep learning data preparation with memory-safe internal sub-batching.
# - Fits MinMax scaler on TRAIN only via a streaming two-pass approach (min/max aggregation).
# - Streams Parquet in moderate row batches.
# - Encodes DL tensors in smaller sub-batches to avoid large allocations.
# - Saves outputs as compressed shard NPZ files (no giant concatenations in RAM).
# - Preserves max_competitor_len and all sequence shapes.

import os
import json
import time
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Configuration and utilities
# -----------------------------

def load_config(config_path: str = None) -> Dict:
    """Load configuration JSON. Exits if not found."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    print(f"--- Loading configuration from: {config_path} ---")
    if not os.path.exists(config_path):
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        raise SystemExit(1)
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ms() -> float:
    return time.time()


def format_secs(sec: float) -> str:
    return f"{sec:.2f} seconds"


# -----------------------------
# Sequence encoding
# -----------------------------

def one_hot_encode_sequence(sequence: str, max_len: int, nucleotide_map: Dict[str, int]) -> np.ndarray:
    """
    One-hot encodes a single sequence into shape (max_len, vocab).
    Uses final index (e.g., 'N') as default for unknowns.
    """
    vocab_size = len(nucleotide_map)
    arr = np.zeros((max_len, vocab_size), dtype=np.float32)
    if not isinstance(sequence, str):
        sequence = ""
    # Convert to RNA alphabet upper-case
    s = sequence.upper().replace('T', 'U')
    default_idx = vocab_size - 1
    upto = min(len(s), max_len)
    for i in range(upto):
        arr[i, nucleotide_map.get(s[i], default_idx)] = 1.0
    return arr


def pad_structure_vectors(json_list: List[str], max_len: int) -> np.ndarray:
    """
    Parse structure_vector JSON strings, pad to max_len, and expand channel dim.
    Returns shape (N, max_len, 1), dtype float32.
    """
    # Convert each value to list via json.loads
    parsed = []
    for j in json_list:
        try:
            v = json.loads(j) if isinstance(j, str) else (j if isinstance(j, list) else [])
        except Exception:
            v = []
        parsed.append(v if isinstance(v, list) else [])
    padded = pad_sequences(parsed, maxlen=max_len, padding='post', dtype='float32')
    return np.expand_dims(padded, axis=-1)


# -----------------------------
# Streaming train/test split
# -----------------------------

def build_assignments(num_rows: int, test_ratio: float, seed: int = 42) -> np.ndarray:
    """
    Build a reproducible train/test assignment array of shape (num_rows,) with values 'train' or 'test'.
    """
    indices = np.arange(num_rows)
    train_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed)
    assignment = np.empty(num_rows, dtype='U5')
    assignment[train_idx] = 'train'
    assignment[test_idx] = 'test'
    return assignment


# -----------------------------
# Streaming MinMax on TRAIN only
# -----------------------------

def stream_minmax_on_train(
    parquet_file: pq.ParquetFile,
    batch_size: int,
    sample_assignment: np.ndarray,
    numerical_features: List[str],
) -> Dict[str, Tuple[float, float]]:
    """
    First pass: compute per-column (min, max) of numerical features, TRAIN rows only, via streaming.
    Returns dict: {feature: (min_val, max_val)}.
    """
    mins = None
    maxs = None
    processed_rows = 0

    total_rows = parquet_file.metadata.num_rows
    total_batches = math.ceil(total_rows / max(1, batch_size))
    print("\nStep 1: Creating group-wise split and fitting the scaler on TRAIN only...")
    t0 = now_ms()

    for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size, columns=numerical_features),
                    total=total_batches, desc="  Scanning TRAIN for min/max"):
        df = batch.to_pandas()
        n = len(df)
        if n == 0:
            continue

        # fetch global indices for this batch
        idx = np.arange(processed_rows, processed_rows + n)
        mask_train = (sample_assignment[idx] == 'train')
        if not np.any(mask_train):
            processed_rows += n
            continue

        df_tr = df.loc[mask_train]
        arr = df_tr[numerical_features].to_numpy(dtype=np.float64)  # float64 for stable min/max
        if arr.size == 0:
            processed_rows += n
            continue

        batch_min = np.nanmin(arr, axis=0)
        batch_max = np.nanmax(arr, axis=0)

        if mins is None:
            mins = batch_min
            maxs = batch_max
        else:
            mins = np.minimum(mins, batch_min)
            maxs = np.maximum(maxs, batch_max)

        processed_rows += n

    if mins is None or maxs is None:
        print("  - WARNING: No TRAIN rows found for numerical feature scaling. Defaulting to [0,1] identity.")
        # Build dummy scaler params
        return {feat: (0.0, 1.0) for feat in numerical_features}

    scaler_params = {}
    for i, feat in enumerate(numerical_features):
        mn = float(mins[i])
        mx = float(maxs[i])
        if not np.isfinite(mn):
            mn = 0.0
        if not np.isfinite(mx):
            mx = 1.0
        if mx == mn:
            # Avoid division by zero; collapse to zeros
            mx = mn + 1.0
        scaler_params[feat] = (mn, mx)

    dt = now_ms() - t0
    print(f"  - Scaler fitted on TRAIN split and saved. Time: {format_secs(dt)}")
    return scaler_params


def apply_minmax_transform(df: pd.DataFrame, features: List[str], scaler_params: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Apply precomputed min/max to df[features] -> returns float32 array in [0,1].
    """
    out = np.empty((len(df), len(features)), dtype=np.float32)
    for j, feat in enumerate(features):
        mn, mx = scaler_params[feat]
        col = df[feat].to_numpy(dtype=np.float32, copy=False)
        rng = (mx - mn)
        if rng == 0.0:  # extra guard
            out[:, j] = 0.0
        else:
            out[:, j] = (col - mn) / rng
    return out


# -----------------------------
# Encoder with internal sub-batching
# -----------------------------

def process_df_to_arrays(
    df_chunk: pd.DataFrame,
    nucleotide_map: Dict[str, int],
    max_primary_len: int,
    max_target_len: int,
    max_competitor_len: int,
    numerical_features: List[str],
    scaler_params: Dict[str, Tuple[float, float]],
    target_feature: str,
    dl_sub_batch_size: int,
    pbar_desc: str,
) -> Tuple[Dict[str, List[np.ndarray]], List[np.ndarray]]:
    """
    Encode df_chunk into model inputs and target arrays in smaller sub-batches.
    Returns dict of lists (each list contains multiple np arrays of that feature) and list of y arrays.
    """
    N = len(df_chunk)
    if N == 0:
        return (
            {
                'primary_sequence_input': [],
                'target_sequence_input': [],
                'competitor_sequence_input': [],
                'primary_structure_input': [],
                'numerical_features_input': []
            },
            []
        )

    X_lists: Dict[str, List[np.ndarray]] = {
        'primary_sequence_input': [],
        'target_sequence_input': [],
        'competitor_sequence_input': [],
        'primary_structure_input': [],
        'numerical_features_input': []
    }
    y_lists: List[np.ndarray] = []

    # Sub-batch encoding
    total_sub = math.ceil(N / max(1, dl_sub_batch_size))
    for start in tqdm(range(0, N, dl_sub_batch_size), total=total_sub, desc=pbar_desc, leave=False):
        sub_df = df_chunk.iloc[start:start + dl_sub_batch_size]

        # Targets
        y_sub = sub_df[target_feature].to_numpy(dtype=np.float32)

        # Sequences
        prim_enc = np.array(
            [one_hot_encode_sequence(s, max_primary_len, nucleotide_map) for s in sub_df['primary_sequence']],
            dtype=np.float32
        )
        targ_enc = np.array(
            [one_hot_encode_sequence(s, max_target_len, nucleotide_map) for s in sub_df['target_sequence']],
            dtype=np.float32
        )
        comp_enc = np.array(
            [one_hot_encode_sequence(s, max_competitor_len, nucleotide_map) for s in sub_df['competitor_sequence']],
            dtype=np.float32
        )

        # Structure vectors
        struct_enc = pad_structure_vectors(sub_df['structure_vector'].tolist(), max_primary_len)  # (n, max_primary_len, 1)

        # Numerical features scaling
        num_scaled = apply_minmax_transform(sub_df, numerical_features, scaler_params)  # (n, F)

        # Append
        X_lists['primary_sequence_input'].append(prim_enc)
        X_lists['target_sequence_input'].append(targ_enc)
        X_lists['competitor_sequence_input'].append(comp_enc)
        X_lists['primary_structure_input'].append(struct_enc)
        X_lists['numerical_features_input'].append(num_scaled)
        y_lists.append(y_sub)

    return X_lists, y_lists


# -----------------------------
# Sharded saving
# -----------------------------

def save_shard_npz(
    out_dir: str,
    prefix: str,  # 'train' or 'test'
    shard_id: int,
    X_lists: Dict[str, List[np.ndarray]],
    y_lists: List[np.ndarray],
    sqrt_target: bool = True
) -> Dict[str, str]:
    """
    Concatenate sub-batches in-memory for this shard only, then save as compressed .npz files.
    Returns map of feature->file_path for this shard (including y).
    """
    ensure_dir(out_dir)
    saved_paths: Dict[str, str] = {}

    # Y
    if y_lists:
        y_arr = np.concatenate(y_lists, axis=0)
        if sqrt_target:
            y_arr = np.sqrt(y_arr, dtype=np.float32, casting='unsafe')
        y_path = os.path.join(out_dir, f"y_{prefix}_shard{shard_id:05d}.npz")
        np.savez_compressed(y_path, data=y_arr)
        saved_paths['y'] = y_path

    # X
    for key, parts in X_lists.items():
        if not parts:
            continue
        arr = np.concatenate(parts, axis=0)
        x_path = os.path.join(out_dir, f"X_{prefix}_{key}_shard{shard_id:05d}.npz")
        np.savez_compressed(x_path, data=arr)
        saved_paths[key] = x_path

    return saved_paths


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    total_start = now_ms()
    print("--- Starting Data Preparation for Deep Learning (Memory-Safe) ---")

    # Load config
    config = load_config()
    params = {**config.get('processing_parameters', {}), **config.get('training_parameters', {})}

    project_root = config['project_root']
    data_root = os.path.join(project_root, config['data_folders']['main_dataset_folder'])
    prepared_folder = os.path.join(data_root, config['data_folders']['prepared_subfolder'])
    output_dl_folder = os.path.join(data_root, config['data_folders']['processed_for_dl_subfolder'])
    ensure_dir(output_dl_folder)

    # Detect latest parquet
    print(f"\nScanning for datasets in: {prepared_folder}")
    parquet_candidates = [f for f in os.listdir(prepared_folder) if f.endswith('.parquet')]
    if not parquet_candidates:
        print("  - FATAL ERROR: No .parquet files found. Please run Stage 1 first.")
        raise SystemExit(1)
    prepared_dataset_filename = sorted(parquet_candidates)[-1]
    prepared_dataset_path = os.path.join(prepared_folder, prepared_dataset_filename)
    print(f"  - Using latest dataset: {prepared_dataset_filename}")

    # Parquet metadata
    parquet_file = pq.ParquetFile(prepared_dataset_path)
    num_rows = parquet_file.metadata.num_rows

    # Parameters
    pad_params = params['sequence_padding']
    max_primary_len = pad_params['max_primary_len']
    max_target_len = pad_params['max_target_len']
    max_competitor_len = pad_params['max_competitor_len']
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    target_feature = params.get('target_feature', 'affinity')
    numerical_features: List[str] = params.get('numerical_features', ['gc_content', 'dg', 'conservation'])

    dp_params = config.get('data_processing', {})
    parquet_batch_size = int(dp_params.get('batch_size_parquet', 10000))  # safer default
    dl_sub_batch_size = int(dp_params.get('dl_sub_batch_size', 50000))    # internal encoder sub-batch size
    test_ratio = float(params.get('test_split_ratio', 0.2))

    # Step A: Build assignments (train/test)
    print("\nStep 0: Planning split and target feature...")
    print(f"  - Total samples (rows): {num_rows}")
    sample_assignment = build_assignments(num_rows, test_ratio, seed=42)
    print(f"  - Train fraction: {(sample_assignment == 'train').sum() / num_rows:.4f}")
    print(f"  - Test fraction:  {(sample_assignment == 'test').sum() / num_rows:.4f}")

    # Step B: Fit scaler on TRAIN via streaming pass
    scaler_start = now_ms()
    scaler_params = stream_minmax_on_train(
        parquet_file=parquet_file,
        batch_size=parquet_batch_size,
        sample_assignment=sample_assignment,
        numerical_features=numerical_features
    )
    # Save scaler params for reproducibility
    scaler_path = os.path.join(output_dl_folder, 'minmax_scaler_params.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  - Scaler parameters saved to: {scaler_path}")
    print(f"  - Time for scaler pass: {format_secs(now_ms() - scaler_start)}")

    # Step C: Process data in memory-safe batches and save shards
    print("\nStep 1: Processing data in memory-safe batches...")
    shard_train = 0
    shard_test = 0
    processed_rows = 0
    total_batches = math.ceil(num_rows / max(1, parquet_batch_size))

    # Manifest for outputs
    manifest = {
        'prepared_dataset': prepared_dataset_filename,
        'num_rows': int(num_rows),
        'train_ratio': float(1.0 - test_ratio),
        'test_ratio': float(test_ratio),
        'numerical_features': numerical_features,
        'sequence_padding': {
            'max_primary_len': int(max_primary_len),
            'max_target_len': int(max_target_len),
            'max_competitor_len': int(max_competitor_len),
        },
        'outputs': {
            'train': [],
            'test': []
        }
    }

    t_batches_start = now_ms()
    for batch in tqdm(parquet_file.iter_batches(batch_size=parquet_batch_size), total=total_batches, desc="  Streaming Parquet batches"):
        df = batch.to_pandas()  # load all columns needed by encoders and numericals

        n = len(df)
        if n == 0:
            continue
        idx = np.arange(processed_rows, processed_rows + n)
        assign = sample_assignment[idx]

        mask_train = (assign == 'train')
        mask_test = (assign == 'test')

        # TRAIN shard for this parquet batch
        if np.any(mask_train):
            df_tr = df.loc[mask_train]
            X_lists_tr, y_lists_tr = process_df_to_arrays(
                df_chunk=df_tr,
                nucleotide_map=nucleotide_map,
                max_primary_len=max_primary_len,
                max_target_len=max_target_len,
                max_competitor_len=max_competitor_len,
                numerical_features=numerical_features,
                scaler_params=scaler_params,
                target_feature=target_feature,
                dl_sub_batch_size=dl_sub_batch_size,
                pbar_desc="    Encoding TRAIN sub-batches"
            )
            shard_train += 1
            saved_tr = save_shard_npz(
                out_dir=output_dl_folder,
                prefix='train',
                shard_id=shard_train,
                X_lists=X_lists_tr,
                y_lists=y_lists_tr,
                sqrt_target=True
            )
            manifest['outputs']['train'].append({
                'shard_id': shard_train,
                'num_samples': int(np.sum([arr.shape[0] for arr in X_lists_tr['primary_sequence_input']]) if X_lists_tr['primary_sequence_input'] else 0),
                'files': saved_tr
            })

        # TEST shard for this parquet batch
        if np.any(mask_test):
            df_te = df.loc[mask_test]
            X_lists_te, y_lists_te = process_df_to_arrays(
                df_chunk=df_te,
                nucleotide_map=nucleotide_map,
                max_primary_len=max_primary_len,
                max_target_len=max_target_len,
                max_competitor_len=max_competitor_len,
                numerical_features=numerical_features,
                scaler_params=scaler_params,
                target_feature=target_feature,
                dl_sub_batch_size=dl_sub_batch_size,
                pbar_desc="    Encoding TEST sub-batches"
            )
            shard_test += 1
            saved_te = save_shard_npz(
                out_dir=output_dl_folder,
                prefix='test',
                shard_id=shard_test,
                X_lists=X_lists_te,
                y_lists=y_lists_te,
                sqrt_target=True
            )
            manifest['outputs']['test'].append({
                'shard_id': shard_test,
                'num_samples': int(np.sum([arr.shape[0] for arr in X_lists_te['primary_sequence_input']]) if X_lists_te['primary_sequence_input'] else 0),
                'files': saved_te
            })

        processed_rows += n

    dt_batches = now_ms() - t_batches_start
    print(f"  - Time for Step 1 (stream + encode + save): {format_secs(dt_batches)}")

    # Step D: Write manifest
    manifest_path = os.path.join(output_dl_folder, 'dl_dataset_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n--- Deep Learning Data Preparation Complete ---")
    print(f"  - Train shards: {shard_train}, Test shards: {shard_test}")
    total_time = now_ms() - total_start
    print(f"Total time taken: {format_secs(total_time)}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Scaler params: {os.path.join(output_dl_folder, 'minmax_scaler_params.json')}")
    print(f"Outputs written to: {output_dl_folder}")


if __name__ == "__main__":
    main()
