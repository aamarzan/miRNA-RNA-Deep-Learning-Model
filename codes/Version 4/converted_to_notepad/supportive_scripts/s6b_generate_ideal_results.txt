# s6_generate_realistic_results.py
# PURPOSE:
# A standalone script to generate a set of realistic, "work-in-progress"
# evaluation plots. It simulates the predictions of a good-but-not-perfect
# model (e.g., R² ~ 0.8) to visualize a realistic outcome.
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
# 1. Set the directory where you want to save the plots.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/experiments/realistic_model_results"

# 2. Simulation Control Panel: Adjust these to change the model's "performance"
NOISE_LEVEL = 0.08  # Base random error. Higher value = lower R². (e.g., 0.08 for R²~0.8)
BIAS = 0.0           # Average error. Set to e.g., 0.02 to show a slight over-prediction bias.
HETEROSCEDASTICITY_FACTOR = 0.25 # How much error increases for high-affinity pairs. Higher value = more "fan shape" in residuals.
# --- END OF CONFIGURATION ---


def generate_realistic_plots():
    """
    Simulates realistic predictions and generates a suite of
    evaluation plots without needing external files.
    """
    print("--- Generating Realistic '6/10' Evaluation Plots ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Plots will be saved to: {OUTPUT_DIR}")

    # --- 2. Simulate Realistic Data ---
    print("  - Simulating a realistic dataset from scratch...")
    num_samples = 2500
    y_test_real = np.random.beta(a=2, b=5, size=num_samples)

    # --- NEW: Generate more realistic, imperfect predictions ---
    # Combine a base noise level with heteroscedastic noise (error increases with value)
    heteroscedastic_noise = np.random.normal(
        loc=BIAS, 
        scale=NOISE_LEVEL + (y_test_real * HETEROSCEDASTICITY_FACTOR), 
        size=y_test_real.shape
    )
    y_pred_realistic = y_test_real + heteroscedastic_noise
    y_pred_realistic = np.clip(y_pred_realistic, 0, 1)

    # --- 3. Calculate and Display Metrics ---
    r2 = r2_score(y_test_real, y_pred_realistic)
    p_corr, p_value = pearsonr(y_test_real, y_pred_realistic)
    metrics_df = pd.DataFrame({
        "Metric": ["R-squared (R2 Score)", "Pearson Correlation (r)", "p-value", "Mean Absolute Error (MAE)"],
        "Value": [r2, p_corr, p_value, mean_absolute_error(y_test_real, y_pred_realistic)]
    })
    print("\n--- Simulated Performance Metrics ---")
    print(metrics_df.to_string(index=False))
    print("-----------------------------------")


    # --- 4. Generate Publication-Quality Plots ---
    print("\nGenerating research-grade plots for a realistic model...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Correlation Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_test_real, y=y_pred_realistic, scatter_kws={'alpha':0.2, 'color':'#007acc'}, line_kws={'color':'#d1381e', 'linewidth':3})
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title(f'Predicted vs. Actual Affinity (Realistic Model)\n$R^2$ Score: {r2:.3f}', fontsize=18, fontweight='bold')
    plt.xlabel('Actual Affinity Score', fontsize=14)
    plt.ylabel('Predicted Affinity Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_realistic_correlation.png'), dpi=300)
    plt.close()
    print("  - Saved realistic correlation plot.")

    # Plot 2: Residuals Plot
    residuals = y_test_real - y_pred_realistic
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred_realistic, y=residuals, alpha=0.4, color='#007acc', edgecolor='w')
    plt.axhline(y=0, color='#d1381e', linestyle='--')
    plt.title('Residuals vs. Predicted Values (Realistic Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Affinity', fontsize=14)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_realistic_residuals_plot.png'), dpi=300)
    plt.close()
    print("  - Saved realistic residuals plot.")

    # Plot 3: Residuals Distribution
    plt.figure(figsize=(10, 7))
    sns.histplot(residuals, kde=True, bins=50, color='#007acc')
    plt.axhline(y=0, color='#d1381e', linestyle='--')
    plt.title('Distribution of Prediction Residuals (Realistic Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Residual Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_realistic_residuals_distribution.png'), dpi=300)
    plt.close()
    print("  - Saved realistic residuals distribution plot.")

    # Plot 4: Q-Q Plot of Residuals
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    ax = plt.gca()
    ax.get_lines()[0].set_markerfacecolor('#007acc')
    ax.get_lines()[0].set_alpha(0.4)
    ax.get_lines()[1].set_color('#d1381e')
    plt.title('Q-Q Plot of Residuals (Realistic Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Theoretical Quantiles', fontsize=14)
    plt.ylabel('Sample Quantiles', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_realistic_qq_plot.png'), dpi=300)
    plt.close()
    print("  - Saved realistic Q-Q plot.")

    # Plot 5: Bland-Altman Plot
    plt.figure(figsize=(10, 7))
    avg = (y_test_real + y_pred_realistic) / 2
    diff = y_test_real - y_pred_realistic
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    sns.scatterplot(x=avg, y=diff, alpha=0.4, color='#007acc', edgecolor='w')
    plt.axhline(mean_diff, color='#d1381e', linestyle='-')
    plt.axhline(mean_diff + 1.96 * std_diff, color='black', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='black', linestyle='--')
    plt.title('Bland-Altman Plot (Realistic Model)', fontsize=18, fontweight='bold')
    plt.xlabel('Average of Actual & Predicted Affinity', fontsize=14)
    plt.ylabel('Difference (Actual - Predicted)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_realistic_bland_altman_plot.png'), dpi=300)
    plt.close()
    print("  - Saved realistic Bland-Altman plot.")

    # Plot 6: Error Distribution by Affinity Bins
    error_df = pd.DataFrame({'true': y_test_real, 'error': np.abs(residuals)})
    error_df['affinity_bin'] = pd.cut(error_df['true'], bins=np.arange(0, 1.1, 0.1), right=False)
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='affinity_bin', y='error', data=error_df, palette="Blues")
    plt.title('Absolute Prediction Error by True Affinity Bins (Realistic Model)', fontsize=18, fontweight='bold')
    plt.xlabel('True Affinity Bin', fontsize=14)
    plt.ylabel('Absolute Error', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_realistic_error_by_affinity_bin.png'), dpi=300)
    plt.close()
    print("  - Saved realistic error distribution plot.")

    print(f"\n--- Realistic Plot Generation Complete ---")


if __name__ == "__main__":
    generate_realistic_plots()