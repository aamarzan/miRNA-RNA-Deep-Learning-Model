import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_distribution_comparison():
    """
    Analyzes a predefined binned distribution and compares it to a theoretical
    normal distribution with the same mean and standard deviation.
    """
    # 1. Manually define the binned data from your output
    data = {
        "[0.3, 0.35)": 5,   "[0.35, 0.4)": 70,  "[0.4, 0.45)": 79,
        "[0.45, 0.5)": 79,  "[0.5, 0.55)": 88,  "[0.55, 0.6)": 91,
        "[0.6, 0.65)": 40,  "[0.65, 0.7)": 1350,"[0.7, 0.75)": 69,
        "[0.75, 0.8)": 64,  "[0.8, 0.85)": 94,  "[0.85, 0.9)": 73,
        "[0.9, 0.95)": 800, "[0.95, 1.0)": 2187
    }
    
    bin_counts = pd.Series(data)
    total_count = bin_counts.sum()
    
    # 2. Calculate approximate mean and std dev from the binned data
    bin_midpoints = [float(label.strip('[)').split(',')[0]) + 0.025 for label in bin_counts.index]
    
    # Weighted Mean
    mean_val = np.sum(bin_midpoints * bin_counts) / total_count
    
    # Weighted Standard Deviation
    variance = np.sum(bin_counts * (bin_midpoints - mean_val)**2) / total_count
    std_dev = np.sqrt(variance)

    print("--- Calculated Statistics from Your Data ---")
    print(f"Approximate Mean: {mean_val:.4f}")
    print(f"Approximate Std Dev: {std_dev:.4f}")

    # 3. Create the plot
    print("\n--- Generating Comparison Plot ---")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))
    
    # Plot the actual distribution as a bar chart
    bin_labels = [label.replace(',', ' to') for label in bin_counts.index]
    ax = sns.barplot(x=bin_labels, y=bin_counts.values, palette='mako', label='Actual Distribution')

    # 4. Overlay the theoretical normal distribution
    # Create a range of x-values for the smooth curve
    x_axis = np.arange(0, 1, 0.01)
    # Calculate the normal distribution PDF, scaled to the total count
    # We need to scale the PDF to match the histogram's height
    scaling_factor = total_count * 0.05 # Total count * bin_width
    y_axis = norm.pdf(x_axis, mean_val, std_dev) * scaling_factor
    
    # Plot on the secondary y-axis to not interfere with bar labels
    ax2 = ax.twinx()
    ax2.plot(x_axis, y_axis, color='crimson', linestyle='--', linewidth=3, label='Theoretical Normal Distribution')
    
    # Formatting
    ax.set_title('Actual Data Distribution vs. Theoretical Normal Distribution', fontsize=18, fontweight='bold')
    ax.set_xlabel('Affinity Score Range', fontsize=14)
    ax.set_ylabel('Number of Samples (Count)', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax2.set_ylabel('Theoretical Density', fontsize=14, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    
    # Create a single legend for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)
    
    plt.tight_layout()
    output_filename = 'normal_distribution_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Success! Comparison chart saved to '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    plot_distribution_comparison()