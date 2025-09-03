# s0f_consolidate_affinity.py (UPGRADED VERSION)
# PURPOSE:
# To read all raw affinity score files, handle duplicates for each miRNA by
# calculating mean, min, and multiple percentiles, and save a separate master
# affinity file for each consolidation method.

import os
import json
import pandas as pd

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

def consolidate_affinity_scores():
    """
    Reads all raw affinity files, aggregates scores by miRNA ID using multiple
    methods, and saves a separate master file for each.
    """
    print("--- Starting Master Affinity Score Consolidation ---")
    config = load_config()

    # 1. Define input and output paths from config
    project_root = config.get('project_root')
    raw_data_folder = os.path.join(project_root, 'dataset', 'raw_data')
    input_folder = os.path.join(raw_data_folder, 'affinity_score', 'good')
    output_folder = os.path.join(raw_data_folder, 'affinity_score', 'select')
    os.makedirs(output_folder, exist_ok=True)
    
    id_col_name = config['data_sources']['affinity']['id_col']
    score_col_name = config['data_sources']['affinity']['score_col']

    # 2. Read and combine all data files
    all_dfs = []
    file_extensions = ('.csv', '.tsv', '.txt')
    score_files = [f for f in os.listdir(input_folder) if f.lower().endswith(file_extensions)]

    if not score_files:
        print(f"❌ ERROR: No score files found in '{input_folder}'")
        return

    print(f"  - Found {len(score_files)} files. Reading and combining data...")
    for filename in score_files:
        filepath = os.path.join(input_folder, filename)
        sep = '\t' if filepath.lower().endswith(('.tsv', '.txt')) else ','
        try:
            df = pd.read_csv(filepath, sep=sep, comment='#', low_memory=False)
            actual_id_col = next((col for col in df.columns if col.lower() == id_col_name.lower()), None)
            actual_score_col = next((col for col in df.columns if col.lower() == score_col_name.lower()), None)

            if actual_id_col and actual_score_col:
                temp_df = df[[actual_id_col, actual_score_col]].copy()
                temp_df.columns = ['miRNA_ID', 'affinity_score']
                all_dfs.append(temp_df)
            else:
                print(f"  - ⚠️ WARNING: Could not find required columns in file: {filename}")
        except Exception as e:
            print(f"  - ⚠️ WARNING: Failed to process file {filename}. Error: {e}")

    if not all_dfs:
        print("❌ ERROR: No data could be loaded. Halting.")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df['affinity_score'] = pd.to_numeric(master_df['affinity_score'], errors='coerce')
    master_df.dropna(inplace=True)
    print(f"  - Combined {len(master_df)} total records from all files.")

    # 3. ⭐ NEW: Define all aggregations, including percentiles
    print("  - Aggregating scores for duplicate miRNAs using multiple methods...")
    
    # Define the percentiles you want to calculate
    #percentiles_to_calc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    percentiles_to_calc = [0.25, 0.75]
    
    # Create a dictionary of aggregation functions
    aggregations = {
        'mean': 'mean',
        'min': 'min'
    }
    for p in percentiles_to_calc:
        # Use a lambda function to calculate each quantile (percentile)
        # The p=p trick is important to correctly capture the value in the lambda
        aggregations[f'p{int(p*100)}'] = lambda x, p=p: x.quantile(p)

    # Calculate all aggregations in a single pass for efficiency
    aggregated_df = master_df.groupby('miRNA_ID')['affinity_score'].agg(**aggregations).reset_index()
    print(f"  - Processed into {len(aggregated_df)} unique miRNA entries.")

    # 4. ⭐ NEW: Loop through the results and save a file for each aggregation
    print("\n--- Saving Master Files for Each Consolidation Method ---")
    
    # Loop through all calculated columns (mean, min, p55, p60, etc.)
    for agg_name in aggregated_df.columns:
        if agg_name == 'miRNA_ID':
            continue # Skip the ID column
            
        # Create a new dataframe for the current aggregation method
        output_df = aggregated_df[['miRNA_ID', agg_name]].copy()
        output_df.rename(columns={'miRNA_ID': id_col_name, agg_name: score_col_name}, inplace=True)
        
        # Define the output path and save the file
        output_filename = f'Master_Affinity_Scores_{agg_name.capitalize()}.txt'
        output_path = os.path.join(output_folder, output_filename)
        output_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
        print(f"  ✅ Saved: {output_filename}")


if __name__ == "__main__":
    consolidate_affinity_scores()