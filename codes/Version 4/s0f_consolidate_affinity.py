# s0f_consolidate_affinity.py
# PURPOSE:
# To read all raw affinity score files, handle duplicates for each miRNA by
# both averaging and taking the minimum score, and save two new master
# affinity files.

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
    Reads all raw affinity files, aggregates scores by miRNA ID using mean and min,
    and saves two separate master files.
    """
    print("--- Starting Master Affinity Score Consolidation ---")
    config = load_config()

    # 1. Define input and output paths from config
    project_root = config.get('project_root')
    raw_data_folder = os.path.join(project_root, 'dataset', 'raw_data')
    
    # User-specified directories
    input_folder = os.path.join(raw_data_folder, 'affinity_score', 'good')
    output_folder = os.path.join(raw_data_folder, 'affinity_score', 'select')
    os.makedirs(output_folder, exist_ok=True)
    
    # Get column names from config for consistency
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
            
            # Find the actual ID and score column names in the dataframe
            actual_id_col = next((col for col in df.columns if col.lower() == id_col_name.lower()), None)
            actual_score_col = next((col for col in df.columns if col.lower() == score_col_name.lower()), None)

            if actual_id_col and actual_score_col:
                # Keep only the necessary columns and standardize their names
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

    # 3. Aggregate the scores by miRNA ID
    print("  - Aggregating scores for duplicate miRNAs...")
    # Use groupby().agg() to calculate mean and min in one pass
    aggregated_df = master_df.groupby('miRNA_ID')['affinity_score'].agg(['mean', 'min']).reset_index()
    print(f"  - Processed into {len(aggregated_df)} unique miRNA entries.")

    # 4. Save the 'Averaging' results
    average_df = aggregated_df[['miRNA_ID', 'mean']].copy()
    average_df.rename(columns={'miRNA_ID': id_col_name, 'mean': score_col_name}, inplace=True)
    avg_output_path = os.path.join(output_folder, 'Master_Affinity_Scores_Average.txt')
    average_df.to_csv(avg_output_path, sep='\t', index=False, float_format='%.6f')
    print(f"✅ Success! Master file with AVERAGED scores saved to:\n   {avg_output_path}")

    # 5. Save the 'Lowest' results
    lowest_df = aggregated_df[['miRNA_ID', 'min']].copy()
    lowest_df.rename(columns={'miRNA_ID': id_col_name, 'min': score_col_name}, inplace=True)
    low_output_path = os.path.join(output_folder, 'Master_Affinity_Scores_Lowest.txt')
    lowest_df.to_csv(low_output_path, sep='\t', index=False, float_format='%.6f')
    print(f"✅ Success! Master file with LOWEST scores saved to:\n   {low_output_path}")


if __name__ == "__main__":
    consolidate_affinity_scores()