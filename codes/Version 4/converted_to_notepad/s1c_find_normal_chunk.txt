# s0g_find_normal_chunk.py
# PURPOSE:
# To analyze all prepared dataset chunks, perform a normality test on the
# affinity scores in each, and rank them to find the chunk with the most
# 'normal-like' distribution.

import os
import json
import pandas as pd
from scipy import stats
from tqdm import tqdm

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

def find_most_normal_chunk():
    """
    Iterates through Parquet files, runs a Shapiro-Wilk test on the affinity
    column, and reports which file's data is closest to a normal distribution.
    """
    print("--- Finding Dataset Chunk with the Most Normal Distribution ---")
    config = load_config()

    # 1. Define paths from config
    project_root = config.get('project_root')
    # The chunks are located in the 'split_parts' sub-directory
    data_folder = os.path.join(project_root, 'dataset', 'prepared_dataset', 'split_parts')
    
    affinity_col_name = config['training_parameters']['target_feature']

    if not os.path.isdir(data_folder):
        print(f"❌ ERROR: Directory not found: {data_folder}")
        return

    parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]
    if not parquet_files:
        print(f"❌ ERROR: No Parquet files found in '{data_folder}'")
        return

    results = []
    print(f"\nAnalyzing {len(parquet_files)} dataset chunks...")
    
    # 2. Iterate through each file and perform the normality test
    # Use tqdm for a progress bar
    for filename in tqdm(parquet_files, desc="  Testing chunks"):
        filepath = os.path.join(data_folder, filename)
        try:
            # Read only the affinity column to save memory
            df = pd.read_parquet(filepath, columns=[affinity_col_name])
            affinity_scores = df[affinity_col_name].dropna()

            # The Shapiro-Wilk test has a limit of 5000 samples for accuracy.
            # We take a random sample if the chunk is larger than that.
            if len(affinity_scores) > 5000:
                affinity_scores = affinity_scores.sample(5000, random_state=42)
            
            # Perform the Shapiro-Wilk test for normality
            shapiro_stat, p_value = stats.shapiro(affinity_scores)
            results.append({'filename': filename, 'shapiro_p_value': p_value})

        except Exception as e:
            print(f"  - ⚠️ WARNING: Could not process file {filename}. Error: {e}")

    if not results:
        print("❌ ERROR: No results could be calculated.")
        return

    # 3. Sort the results to find the best chunk
    # A higher p-value indicates the data is more likely to be normally distributed.
    sorted_results = sorted(results, key=lambda x: x['shapiro_p_value'], reverse=True)

    # 4. Display the ranked list
    print("\n--- Normality Test Results (Ranked by Shapiro-Wilk p-value) ---")
    print("A higher p-value means the distribution is closer to a perfect normal distribution.")
    
    results_df = pd.DataFrame(sorted_results)
    print(results_df.to_string(index=False))
    
    print("\n--- Recommendation ---")
    print(f"The chunk with the most 'perfect' normal distribution is:")
    print(f"✅ {sorted_results[0]['filename']} (p-value: {sorted_results[0]['shapiro_p_value']:.4f})")


if __name__ == "__main__":
    find_most_normal_chunk()