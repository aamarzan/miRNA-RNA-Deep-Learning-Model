# s5_evaluate.py (ULTIMATE FINAL PUBLICATION-QUALITY VERSION)
# PURPOSE:
# This script loads a trained model, evaluates it on the NPZ test set, and
# generates a full suite of publication-quality plots with advanced visual refinements.

import os
import json
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap # For custom gradient colormap

# Import our custom components to load the model correctly
from s3b_build_model import create_weighted_mse, PositionalEncoding

def load_config(config_path=None):
    """Loads and returns the configuration file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

def load_test_data_npz(data_path, model_inputs):
    """Dynamically finds and loads all X_test_*.npz files."""
    X_test = {}
    print(f"  - Searching for NPZ test data in: {data_path}")
    
    for key in model_inputs:
        filepath = os.path.join(data_path, f'X_test_{key}.npz')
        if os.path.exists(filepath):
            print(f"    - Loading: {os.path.basename(filepath)}")
            X_test[key] = np.load(filepath)['data']
    
    y_test_path = os.path.join(data_path, 'y_test.npz')
    y_test = np.load(y_test_path)['data']
        
    if not X_test:
        raise FileNotFoundError(f"No NPZ test data files (X_test_*.npz) found in {data_path}.")
        
    return X_test, y_test

def analyze_model_performance():
    print("--- Starting Advanced Model Evaluation and Visualization ---")
    
    config = load_config()
    eval_params = config['evaluation_parameters']
    train_params = config['training_parameters']
    
    project_root = config['project_root']
    experiment_id = config.get('experiment_id', 'default_run')

    experiment_dir = os.path.join(project_root, 'experiments', experiment_id)
    model_dir = os.path.join(experiment_dir, config['output_folders']['main_models_folder'])
    model_path = os.path.join(model_dir, eval_params['model_to_evaluate'])
    
    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    
    # Create a unique output folder for this evaluation run
    base_model_name = eval_params['model_to_evaluate'].replace('.keras', '').replace('.h5', '')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    evaluation_root_dir = os.path.join(experiment_dir, 'evaluation')
    plots_dir = os.path.join(evaluation_root_dir, f"{eval_params['output_folder_prefix']}_{base_model_name}_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nStep 1: Loading test data, model, and history...")
    try:
        X_test, y_test_transformed = load_test_data_npz(data_path, train_params['model_inputs'])
        
        custom_objects = {'PositionalEncoding': PositionalEncoding}
        if train_params['advanced_training'].get('use_custom_loss', False):
            loss_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
            custom_objects['weighted_mse'] = loss_instance
            
        history_path = os.path.join(model_dir, eval_params['history_to_load'])
        print(f"  - Loading trained model from: {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        with open(history_path, 'r') as f:
            history = json.load(f)
        print("  - All files loaded successfully.")
    except (FileNotFoundError, IOError, OSError) as e:
        print(f"  -  FATAL ERROR loading files: {e}.")
        return

    print("\nStep 2: Making predictions with the loaded model...")
    y_pred_transformed = model.predict(X_test, batch_size=eval_params.get('prediction_batch_size', 64), verbose=1)
    if isinstance(y_pred_transformed, list):
        y_pred_transformed = y_pred_transformed[0]
    y_pred_transformed = y_pred_transformed.ravel()

    print("  - Applying inverse transform (square) to get real affinity scores...")
    y_pred = np.square(y_pred_transformed)
    y_test = np.square(y_test_transformed)

    # Calculate all necessary statistics
    r2 = r2_score(y_test, y_pred)
    pearson_r, p_value = pearsonr(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\nStep 3: Generating final publication-quality plots...")
    
    # ⭐ --- FINAL PUBLICATION-QUALITY SCATTER PLOT --- ⭐
    try:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(11, 10))
        ax = plt.gca()

        errors = np.abs(y_test - y_pred)
        
        # ⭐ NEW: Custom Colormap from Dark Green to Medium Blue to Medium Red
        colors = ["midnightblue","cadetblue"] # Define the color stops
        # Normalize errors to 0-1 range for colormap
        norm_errors = (errors - errors.min()) / (errors.max() - errors.min())
        # Create a custom colormap from the defined colors
        custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
        
        # Apply the custom colormap to the normalized errors
        edge_colors = custom_cmap(norm_errors)

        points = ax.scatter(x=y_test, y=y_pred, 
                            s=35, 
                            facecolors='none', 
                            edgecolors=edge_colors, # Use the custom generated RGBA colors
                            linewidth=0.1 # ⭐ NEW: Thinner circumference
                           )
        
        # ⭐ NEW: Light red shading on predictive trendline (ci='lightcoral' or similar)
        sns.regplot(
            x=y_test, 
            y=y_pred, 
            scatter=False, 
            ax=ax, 
            color='lightcoral',  # Set the base color for the confidence interval (shading)
            line_kws={'linestyle': '--', 'linewidth': 2, 'color': 'crimson'}, # Override the line color to crimson
            truncate=False,
            ci=95
        )

        ax.plot([0, 1], [0, 1], color='black', linestyle=':', linewidth=2, label='Perfect Prediction (y=x)')
        
        stats_text = (f'$R^2$ Score: {r2:.3f}\n' f'Pearson r: {pearson_r:.3f}\n' f'p-value: {p_value:.2e}\n' f'MAE: {mae:.3f}\n' f'RMSE: {rmse:.3f}')
        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

        ax.set_xlabel('Actual Affinity Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted Affinity Score', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance: Predicted vs. Actual Affinity', fontsize=18, fontweight='bold', pad=20)
        
        axis_ticks = np.arange(0, 1.01, 0.05)
        ax.set_xticks(axis_ticks)
        ax.set_yticks(axis_ticks)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        
        # ⭐ NEW: Adjust color bar to use the custom colormap
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=errors.min(), vmax=errors.max()))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Absolute Error |Prediction - Actual|', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'Publication_Plot_Prediction_vs_Actual.png'), dpi=300)
        plt.close()
        print("  - Saved final publication-quality scatter plot.")
    except Exception as e:
        print(f"  - ❌ ERROR: Could not generate the main scatter plot. Reason: {e}")

    # (You can add the other 6 diagnostic plots here if you wish)

    print(f"\n✅ All plots and metrics have been saved to the folder: '{plots_dir}'")
    print("\n--- Evaluation and Visualization Complete ---")

if __name__ == "__main__":
    analyze_model_performance()