# s0d_analyze_affinity_distribution.py (UPGRADED VERSION 3)
# PURPOSE:
# To read all affinity score files, group them into bins of width 0.05 up to 1.0,
# and display the distribution as a summary table and a detailed bar chart.

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def analyze_affinity_distribution(bin_width=0.05):
    """
    Loads all affinity scores, groups them into bins, and saves a plot of the distribution.
    """
    print("--- Analyzing Binned Distribution of Affinity Scores ---")
    config = load_config()

    # 1. Get paths and column names from config
    project_root = config.get('project_root')
    raw_data_folder = os.path.join(project_root, 'dataset', 'raw_data')
    affinity_folder = os.path.join(raw_data_folder, config['data_sources']['affinity']['folder'])
    score_col_name = config['data_sources']['affinity']['score_col']

    # 2. Load all scores into a single list
    all_scores = []
    score_files = [f for f in os.listdir(affinity_folder) if f.lower().endswith(('.csv', '.tsv', '.txt'))]

    if not score_files:
        print(f"❌ ERROR: No score files found in the directory: {affinity_folder}")
        return

    print(f"  - Found {len(score_files)} score file(s). Reading data...")
    for filename in score_files:
        filepath = os.path.join(affinity_folder, filename)
        sep = '\t' if filepath.lower().endswith(('.tsv', '.txt')) else ','
        try:
            df = pd.read_csv(filepath, sep=sep, comment='#', low_memory=False)
            actual_col = next((col for col in df.columns if col.lower() == score_col_name.lower()), None)
            if actual_col:
                df.dropna(subset=[actual_col], inplace=True)
                all_scores.extend(pd.to_numeric(df[actual_col], errors='coerce').dropna().tolist())
            else:
                 print(f"  - ⚠️ WARNING: Expected score column '{score_col_name}' not found in file {filename}.")
        except Exception as e:
            print(f"  - ⚠️ WARNING: Could not process file {filename}. Error: {e}")

    if not all_scores:
        print("❌ ERROR: No valid scores could be loaded from the files.")
        return

    # 3. Group the scores into bins
    scores_series = pd.Series(all_scores)
    
    # ⭐ NEW: Create bins from 0 to 1.0 with the specified width
    # The stop parameter is 1.0 + bin_width to ensure the 1.0 value is included
    bins = np.arange(start=0, stop=1.0 + bin_width, step=bin_width)
    
    binned_scores = pd.cut(scores_series, bins=bins, right=False)
    
    # Calculate the number of scores in each bin
    bin_counts = binned_scores.value_counts().sort_index()

    # Note any scores that fall outside the 0-1 range
    outliers = scores_series[scores_series > 1.0]
    
    # 4. Display the results table
    print(f"\n--- Affinity Score Distribution (Bin Width: {bin_width}) ---")
    print(f"Total number of scores analyzed: {len(all_scores)}")
    if not outliers.empty:
        print(f"  - NOTE: Found {len(outliers)} scores with a value > 1.0; they are excluded from the summary below.")
    print(bin_counts.to_string())

    # 5. Create and save a bar chart visualization
    print("\n  - Generating bar chart...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8)) # Made figure wider for more bins
    
    plot_data = bin_counts.reset_index()
    plot_data.columns = ['Affinity Range', 'Count']
    plot_data['Affinity Range'] = plot_data['Affinity Range'].astype(str)
    
    ax = sns.barplot(x='Affinity Range', y='Count', data=plot_data, palette='mako')
    
    plt.title('Distribution of Affinity Scores (0 to 1.0)', fontsize=18, fontweight='bold')
    plt.xlabel('Affinity Score Range', fontsize=14)
    plt.ylabel('Number of Samples (Count)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of each bar for clarity
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    output_filename = 'affinity_score_distribution_detailed.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Success! Chart saved to '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    # ⭐ NEW: The bin_width is now set to 0.05
    analyze_affinity_distribution(bin_width=0.05)