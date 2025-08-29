# s5_evaluate.py (Final Version with Publication-Quality Plots & .npz Support)
#
# PURPOSE:
# This script evaluates the performance of a trained model on the held-out test set.
# It calculates key regression metrics (R², MAE, etc.) and generates a suite of
# publication-quality plots to visualize the model's accuracy and behavior.
#
# WORKFLOW:
# Run this script after Stage 3 is complete. Ensure the 'evaluation_parameters'
# in config.json point to the correct model and history files from your training run.
#
import os
import json
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom components to load the model correctly
from s3b_build_model import create_weighted_mse, PositionalEncoding

def load_config(config_path=None):
    """
    Loads the configuration from a JSON file.
    If no path is given, it automatically finds 'config.json' in the same directory as the script.
    """
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

def load_test_data(data_path):
    """Dynamically finds and loads all X_test_*.npz files."""
    X_test = {}
    print(f"  - Searching for test data in: {data_path}")
    
    all_files = os.listdir(data_path)
    test_files = sorted([f for f in all_files if f.startswith('X_test_') and f.endswith('.npz')])
    
    if not test_files:
        raise FileNotFoundError(f"No test data files (.npz) found in {data_path}. Please complete Stage 2.")

    for f in test_files:
        key = f.replace('X_test_', '').replace('.npz', '')
        print(f"    - Loading: {f}")
        with np.load(os.path.join(data_path, f), mmap_mode='r') as loaded_file:
            X_test[key] = loaded_file['data']
    
    with np.load(os.path.join(data_path, 'y_test.npz'), mmap_mode='r') as loaded_file:
        y_test = loaded_file['data']
        
    return X_test, y_test

def analyze_model_performance():
    print("--- Starting Model Evaluation and Visualization ---")
    
    config = load_config()
    eval_params = config['evaluation_parameters']
    train_params = config['training_parameters']
    
    project_root = config['project_root']
    experiment_id = config.get('experiment_id', 'default_run') # Use the same ID as in training

    # Point to the specific experiment folder for inputs and outputs
    experiment_dir = os.path.join(project_root, 'experiments', experiment_id)
    model_dir = os.path.join(experiment_dir, config['output_folders']['main_models_folder'])
    model_name_from_config = eval_params['model_to_evaluate']
    base_model_name = model_name_from_config.replace('.keras', '').replace('.h5', '')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    # New folder will be inside an 'evaluation' subfolder, e.g., experiments/run_id/evaluation/eval_plots_model_name_timestamp
    evaluation_root_dir = os.path.join(experiment_dir, 'evaluation')
    plots_dir = os.path.join(evaluation_root_dir, f"{eval_params['output_folder_prefix']}_{base_model_name}_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    
    print("\nStep 1: Loading test data, model, and history...")
    try:
        X_test, y_test = load_test_data(data_path)
        
        custom_objects = {}
        if train_params['advanced_training']['use_custom_loss']:
            loss_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
            custom_objects['weighted_mse'] = loss_instance
        custom_objects['PositionalEncoding'] = PositionalEncoding
            
        print("  - Loading model with custom objects...")
        model_path = os.path.join(model_dir, eval_params['model_to_evaluate'])
        history_path = os.path.join(model_dir, eval_params['history_to_load'])

        print(f"  - Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        print(f"  - Loading history from: {history_path}")
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        print("  - All files loaded successfully.")
    except (FileNotFoundError, IOError) as e:
        print(f"  - FATAL ERROR loading files: {e}.")
        return

    print("\nStep 2: Evaluating model performance on the test set...")
    y_pred_transformed = model.predict(X_test, batch_size=eval_params.get('prediction_batch_size', 1024), verbose=1).ravel()

    # --- NEW: Apply inverse transform (square) to get real affinity scores ---
    print("  - Applying inverse transform (square) to predictions and test labels...")
    y_pred = np.square(y_pred_transformed)
    y_test = np.square(y_test) # The y_test loaded from .npz is also on the sqrt scale

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    p_corr, _ = pearsonr(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": ["R-squared (R2 Score)", "Pearson Correlation (r)", "Mean Squared Error (MSE)", "Mean Absolute Error (MAE)"],
        "Value": [r2, p_corr, mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)]
    })
    
    print("\n--- Performance Metrics Summary ---")
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    metrics_df.to_csv(os.path.join(plots_dir, 'performance_metrics.csv'), index=False, float_format='%.4f')
    print(f"\n  - Metrics table saved to: '{plots_dir}'")

    print("\nStep 3: Generating publication-quality plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Correlation Scatter Plot with Density Hexbins
    sample_indices = np.random.choice(len(y_test), size=min(10000, len(y_test)), replace=False)
    g = sns.jointplot(x=y_test[sample_indices], y=y_pred[sample_indices], kind='hex', cmap='viridis', gridsize=50)
    g.ax_joint.plot([0, 1], [0, 1], color='white', linestyle='--', linewidth=2, label='Perfect Prediction')
    g.ax_joint.set_xlabel('Actual Affinity Score', fontsize=12, fontweight='bold')
    g.ax_joint.set_ylabel('Predicted Affinity Score', fontsize=12, fontweight='bold')
    plt.suptitle(f'Predicted vs. Actual Affinity\n$R^2$ Score: {r2:.3f} | Pearson r: {p_corr:.3f}', y=1.03, fontsize=16, fontweight='bold')
    g.ax_joint.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_correlation_density.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.close()
    print("  - Saved enhanced correlation plot.")

    # Plot 2: Training History
    plt.figure(figsize=(12, 7))
    plt.plot(history['loss'], label='Training Loss', color='darkblue', lw=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--', lw=2)
    plt.title('Model Loss Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend(fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300)
    plt.close()
    print("  - Generating advanced diagnostic plots...")
    residuals = y_test - y_pred
    
    # Plot 3: Residuals Plot
    # Shows if errors are randomly distributed or have a pattern.
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted Values', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Affinity', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_plot.png'), dpi=300)
    plt.close()
    print("  - Saved residuals plot.")

    # Plot 4: Residuals Distribution
    # Shows if the model has a bias (e.g., consistently over/under-predicting).
    plt.figure(figsize=(10, 7))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title('Distribution of Prediction Residuals', fontsize=16, fontweight='bold')
    plt.xlabel('Residual Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_distribution.png'), dpi=300)
    plt.close()
    print("  - Saved residuals distribution plot.")

    # Plot 5: Q-Q Plot of Residuals
    # A rigorous check for the normality of errors.
    plt.figure(figsize=(7, 7))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=16, fontweight='bold')
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Sample Quantiles', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'qq_plot.png'), dpi=300)
    plt.close()
    print("  - Saved Q-Q plot.")

    # Plot 6: Bland-Altman Plot
    # Visualizes the agreement between predictions and actual values.
    plt.figure(figsize=(10, 7))
    avg = (y_test + y_pred) / 2
    diff = y_test - y_pred
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    sns.scatterplot(x=avg, y=diff, alpha=0.5)
    plt.axhline(mean_diff, color='r', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
    plt.title('Bland-Altman Plot for Prediction Agreement', fontsize=16, fontweight='bold')
    plt.xlabel('Average of Actual and Predicted Affinity', fontsize=12)
    plt.ylabel('Difference (Actual - Predicted)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bland_altman_plot.png'), dpi=300)
    plt.close()
    print("  - Saved Bland-Altman plot.")

    # Plot 7: Error Distribution by Affinity Bins
    # Checks if the model is better at predicting high vs. low affinity.
    error_df = pd.DataFrame({'true': y_test, 'error': np.abs(residuals)})
    error_df['affinity_bin'] = pd.cut(error_df['true'], bins=np.arange(0, 1.1, 0.1), right=False)
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='affinity_bin', y='error', data=error_df)
    plt.title('Absolute Prediction Error by True Affinity Bins', fontsize=16, fontweight='bold')
    plt.xlabel('True Affinity Bin', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_by_affinity_bin.png'), dpi=300)
    plt.close()
    print("  - Saved error distribution plot.")
    
    
    print(f"\n  - All plots and metrics have been saved to the folder: '{plots_dir}'")
    print("\n--- Evaluation and Visualization Complete ---")

if __name__ == "__main__":
    analyze_model_performance()