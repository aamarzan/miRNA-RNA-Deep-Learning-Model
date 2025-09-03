# show_colors.py
# PURPOSE:
# A simple script to generate a visual chart of all named colors
# available in the Matplotlib library for easy reference.

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_color_chart():
    """
    Creates and saves an image displaying all of Matplotlib's named colors.
    """
    print("--- Generating Matplotlib Color Chart ---")

    # Get the dictionary of color names and their hex codes
    colors = mcolors.CSS4_COLORS
    # Sort colors by hue for a more organized chart
    sorted_colors = sorted(colors, key=lambda c: mcolors.to_rgb(c))
    
    n_colors = len(sorted_colors)
    n_cols = 4  # Number of columns in the plot
    n_rows = n_colors // n_cols + 1

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, n_rows * 0.5))
    ax.set_title('Matplotlib Named Colors Reference Chart', fontsize=18, fontweight='bold')

    # Iterate and plot each color with its name
    for i, color_name in enumerate(sorted_colors):
        row = i % n_rows
        col = i // n_rows
        y = row
        x_start = col * 2
        
        ax.text(x_start + 0.9, y, color_name, fontsize=10, ha='right', va='center')
        ax.add_patch(
            plt.Rectangle((x_start + 1, y - 0.4), 0.8, 0.8, color=color_name)
        )

    # Format the plot to be clean and readable
    ax.set_xlim(0, n_cols * 2)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.axis('off') # Hide axes and spines

    plt.tight_layout()
    output_filename = 'matplotlib_named_colors.png'
    plt.savefig(output_filename, dpi=300)
    
    print(f"✅ Success! Color chart saved to '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    plot_color_chart()