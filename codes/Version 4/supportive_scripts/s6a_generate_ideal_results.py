# s6_generate_ideal_results.py
# PURPOSE:
# A standalone script to generate a set of ideal, publication-quality evaluation
# plots. It simulates the predictions of a hypothetical, high-performance model
# to visualize the project's target performance (R² > 0.97).
# This script is self-contained and does not require any external data files.
#
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# --- ⚙️ USER CONFIGURATION ---
# Set the directory where you want to save the ideal plots.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/experiments/ideal_model_results_final"
# --- END OF CONFIGURATION ---


def generate_ideal_plots():
    """
    Simulates near-perfect predictions and generates a suite of
    ideal evaluation plots without needing external files.
    """
    print("--- Generating Ideal Evaluation Plots for a Future Super Model ---")

    # --- 1. Create Output Directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Plots will be saved to: {OUTPUT_DIR}")

    # --- 2. Simulate High-Quality Data ---
    print("  - Simulating a high-quality dataset from scratch...")
    # Simulation Parameters (tweak these to change the visuals)
    num_samples = 2500
    # Use a Beta distribution for realistic affinity scores (skewed towards lower values)
    a, b = 2, 5 
    noise_scale = 0.03 # Lower value = higher R²

    # Generate the "real" test data
    y_test_real = np.random.beta(a, b, size=num_samples)

    # Generate the "ideal" predictions by adding a small amount of unbiased noise
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=y_test_real.shape)
    y_pred_ideal = y_test_real + noise
    y_pred_ideal = np.clip(y_pred_ideal, 0, 1) # Ensure predictions are in the valid 0-1 range

    # --- 3. Calculate and Display Ideal Metrics ---
    r2 = r2_score(y_test_real, y_pred_ideal)
    p_corr, p_value = pearsonr(y_test_real, y_pred_ideal)
    metrics_df = pd.DataFrame({
        "Metric": ["R-squared (R2 Score)", "Pearson Correlation (r)", "p-value", "Mean Absolute Error (MAE)"],
        "Value": [r2, p_corr, p_value, mean_absolute_error(y_test_real, y_pred_ideal)]
    })
    print("\n--- Simulated Performance Metrics ---")
    print(metrics_df.to_string(index=False))
    print("-----------------------------------")


    # --- 4. Generate Publication-Quality Plots ---
    print("\nGenerating premium, research-grade plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Correlation Scatter Plot with Regression Line
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_test_real, y=y_pred_ideal, 
                scatter_kws={'alpha':0.2, 'color':'#007acc'}, 
                line_kws={'color':'#d1381e', 'linewidth':3})
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title(f'Predicted vs. Actual Affinity (Ideal Model)\n$R^2$ Score: {r2:.3f}', fontsize=18, fontweight='bold')
    plt.xlabel('Actual Affinity Score', fontsize=14)
    plt.ylabel('Predicted Affinity Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_ideal_correlation.png'), dpi=300)
    plt.close()
    print("  - Saved ideal correlation plot.")

    # Plot 2: Residuals Plot
    residuals = y_test_real - y_pred_ideal
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred_ideal, y=residuals, alpha=0.4, color='#007acc', edgecolor='w')
    plt.axhline(y=0, color='#d1381e', linestyle='--')
    plt.title('Residuals vs. Predicted Values (Ideal Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Affinity', fontsize=14)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
    plt.ylim(min(residuals.min(), -0.15), max(residuals.max(), 0.15))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_ideal_residuals_plot.png'), dpi=300)
    plt.close()
    print("  - Saved ideal residuals plot.")

    # Plot 3: Residuals Distribution
    plt.figure(figsize=(10, 7))
    sns.histplot(residuals, kde=True, bins=50, color='#007acc')
    plt.axvline(x=0, color='#d1381e', linestyle='--')
    plt.title('Distribution of Prediction Residuals (Ideal Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Residual Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_ideal_residuals_distribution.png'), dpi=300)
    plt.close()
    print("  - Saved ideal residuals distribution plot.")

    # Plot 4: Q-Q Plot of Residuals
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    
    # --- FIX: Get the current axes (gca) before modifying lines ---
    ax = plt.gca() 
    ax.get_lines()[0].set_markerfacecolor('#007acc')
    ax.get_lines()[0].set_alpha(0.4)
    ax.get_lines()[1].set_color('#d1381e')
    
    plt.title('Q-Q Plot of Residuals (Ideal Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Theoretical Quantiles', fontsize=14)
    plt.ylabel('Sample Quantiles', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_ideal_qq_plot.png'), dpi=300)
    plt.close()
    print("  - Saved ideal Q-Q plot.")

    # Plot 5: Bland-Altman Plot
    plt.figure(figsize=(10, 7))
    avg = (y_test_real + y_pred_ideal) / 2
    diff = y_test_real - y_pred_ideal
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    sns.scatterplot(x=avg, y=diff, alpha=0.4, color='#007acc', edgecolor='w')
    plt.axhline(mean_diff, color='#d1381e', linestyle='-')
    plt.axhline(mean_diff + 1.96 * std_diff, color='black', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='black', linestyle='--')
    plt.title('Bland-Altman Plot (Ideal Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Average of Actual & Predicted Affinity', fontsize=14)
    plt.ylabel('Difference (Actual - Predicted)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_ideal_bland_altman_plot.png'), dpi=300)
    plt.close()
    print("  - Saved ideal Bland-Altman plot.")

    # Plot 6: Error Distribution by Affinity Bins
    error_df = pd.DataFrame({'true': y_test_real, 'error': np.abs(residuals)})
    error_df['affinity_bin'] = pd.cut(error_df['true'], bins=np.arange(0, 1.1, 0.1), right=False)
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='affinity_bin', y='error', data=error_df, palette="Blues")
    plt.title('Absolute Prediction Error by True Affinity Bins (Ideal Model)', fontsize=18, fontweight='bold')
    plt.xlabel('True Affinity Bin', fontsize=14)
    plt.ylabel('Absolute Error', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_ideal_error_by_affinity_bin.png'), dpi=300)
    plt.close()
    print("  - Saved ideal error distribution plot.")

    print(f"\n--- Ideal Plot Generation Complete ---")


if __name__ == "__main__":
    generate_ideal_plots()
    
