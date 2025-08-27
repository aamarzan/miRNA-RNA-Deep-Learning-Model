# s5_evaluate.py (Final Version with Publication-Quality Plots)
import os
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from s3_build_model import create_weighted_mse

def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
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
    for f in test_files:
        key = f.replace('X_test_', '').replace('.npz', '')
        print(f"    - Loading: {f}")
        X_test[key] = np.load(os.path.join(data_path, f), mmap_mode='r')
    
    y_test = np.load(os.path.join(data_path, 'y_test.npz'), mmap_mode='r')
    return X_test, y_test

def analyze_model_performance():
    print("--- Starting Model Evaluation and Visualization ---")
    
    config = load_config()
    eval_params = config['evaluation_parameters']
    train_params = config['training_parameters']
    
    project_root = config['project_root']
    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    model_dir = os.path.join(project_root, config['output_folders']['main_models_folder'])
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plots_dir = os.path.join(model_dir, f"{eval_params['output_folder_prefix']}_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nStep 1: Loading test data, model, and history...")
    try:
        X_test, y_test = load_test_data(data_path)
        
        custom_objects = {}
        if train_params['advanced_training']['use_custom_loss']:
            loss_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
            custom_objects = {'weighted_mse': loss_instance}
            print("  - Loading model with custom weighted MSE loss.")
            
        model = tf.keras.models.load_model(os.path.join(model_dir, eval_params['model_to_evaluate']), custom_objects=custom_objects)
        
        with open(os.path.join(model_dir, eval_params['history_to_load']), 'r') as f:
            history = json.load(f)
            
        print("  - All files loaded successfully.")
    except (FileNotFoundError, IOError) as e:
        print(f"  - FATAL ERROR loading files: {e}.")
        return

    print("\nStep 2: Evaluating model performance on the test set...")
    y_pred = model.predict(X_test, batch_size=eval_params.get('prediction_batch_size', 1024), verbose=1).ravel()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    p_corr, _ = pearsonr(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R-squared (R2 Score)", "Pearson Correlation (r)"],
        "Value": [mse, mae, r2, p_corr]
    })
    
    print("\n--- Performance Metrics Summary ---")
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    
    metrics_df.to_csv(os.path.join(plots_dir, 'performance_metrics.csv'), index=False, float_format='%.4f')
    print(f"\n  - Metrics table saved to: '{plots_dir}'")

    print("\nStep 3: Generating publication-quality plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Enhanced Correlation Scatter Plot with Density
    plt.figure(figsize=(10, 10))
    sample_indices = np.random.choice(len(y_test), size=min(10000, len(y_test)), replace=False)
    g = sns.jointplot(x=y_test[sample_indices], y=y_pred[sample_indices], kind='hex', cmap='viridis', gridsize=50)
    g.ax_joint.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    g.ax_joint.set_xlabel('Actual Affinity Score', fontsize=12)
    g.ax_joint.set_ylabel('Predicted Affinity Score', fontsize=12)
    plt.suptitle(f'Predicted vs. Actual Affinity\n$R^2$ Score: {r2:.3f} | Pearson r: {p_corr:.3f}', y=1.02, fontsize=16, fontweight='bold')
    g.ax_joint.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_correlation_density.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Saved enhanced correlation plot.")

    # Plot 2: Enhanced Residuals Plot
    residuals = y_test[sample_indices] - y_pred[sample_indices]
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=y_pred[sample_indices], y=residuals, alpha=0.5, color='royalblue')
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.title('Residuals (Actual - Predicted) vs. Predicted Value', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Affinity Score', fontsize=12)
    plt.ylabel('Residual (Error)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_plot.png'), dpi=300)
    plt.close()
    print("  - Saved residuals plot.")

    # Plot 3: Training History
    plt.figure(figsize=(12, 7))
    plt.plot(history['loss'], label='Training Loss', color='darkblue', lw=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--', lw=2)
    plt.title('Model Loss (Mean Squared Error) Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend(fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300)
    plt.close()
    print("  - Saved training history plot.")

    print(f"\n  - All plots have been saved to the folder: '{plots_dir}'")
    print("\n--- Evaluation and Visualization Complete ---")

if __name__ == "__main__":
    analyze_model_performance()