# s0h_visualize_all_chunks.py (MODIFIED)
# PURPOSE:
# To analyze ALL prepared dataset chunks and generate a detailed affinity
# distribution plot for each one.

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats # Keep stats for potential future use

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

def plot_binned_distribution(scores_series, filename, output_dir):
    """
    Generates and saves a binned bar chart for a given series of scores.
    """
    bin_width = 0.05
    bins = np.arange(start=0, stop=1.0 + bin_width, step=bin_width)
    binned_scores = pd.cut(scores_series, bins=bins, right=False)
    bin_counts = binned_scores.value_counts().sort_index()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))
    
    plot_data = bin_counts.reset_index()
    plot_data.columns = ['Affinity Range', 'Count']
    plot_data['Affinity Range'] = plot_data['Affinity Range'].astype(str)
    
    ax = sns.barplot(x='Affinity Range', y='Count', data=plot_data, palette='viridis')
    
    plt.title(f'Affinity Score Distribution for:\n{filename}', fontsize=18, fontweight='bold')
    plt.xlabel('Affinity Score Range', fontsize=14)
    plt.ylabel('Number of Samples (Count)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    safe_filename = filename.replace('.parquet', '.png')
    output_path = os.path.join(output_dir, f"distribution_{safe_filename}")
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_all_chunks():
    """
    Finds ALL chunks and creates a distribution plot for each one.
    """
    print(f"--- Visualizing Affinity Distributions for ALL Chunks ---")
    config = load_config()

    # 1. Define paths from config
    project_root = config.get('project_root')
    data_folder = os.path.join(project_root, 'dataset', 'prepared_dataset', 'split_parts')
    affinity_col_name = config['training_parameters']['target_feature']
    
    output_dir = os.path.join(project_root, 'dataset', 'prepared_dataset', 'distribution_plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"  - Plots will be saved to: {output_dir}")

    # 2. Get the list of all chunks
    parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]
    if not parquet_files:
        print(f"❌ ERROR: No Parquet files found in '{data_folder}'")
        return

    # 3. ⭐ MODIFICATION: Generate a plot for every file found
    print(f"\nGenerating distribution plots for all {len(parquet_files)} chunks...")
    for filename in tqdm(parquet_files, desc="  Generating plots"):
        filepath = os.path.join(data_folder, filename)
        try:
            df = pd.read_parquet(filepath, columns=[affinity_col_name])
            affinity_scores = df[affinity_col_name].dropna()
            plot_binned_distribution(affinity_scores, filename, output_dir)
        except Exception as e:
            print(f"  - ⚠️ WARNING: Could not generate plot for {filename}. Error: {e}")
            
    print(f"\n✅ Success! All {len(parquet_files)} plots have been saved.")


if __name__ == "__main__":
    visualize_all_chunks()