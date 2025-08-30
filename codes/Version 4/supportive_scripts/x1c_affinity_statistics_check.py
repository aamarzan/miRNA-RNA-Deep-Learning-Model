import pandas as pd

# --- CONFIGURATION ---
AFFINITY_FILE_PATH = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/interactions_human.microT.mirbase.txt"
AFFINITY_COLUMN_NAME = "interaction_score"
SEPARATOR = '\t' # Use '\t' for tab-separated files, ',' for comma-separated
# --- END CONFIGURATION ---

try:
    df = pd.read_csv(AFFINITY_FILE_PATH, sep=SEPARATOR, comment='#')
    
    # --- NEW: Automatically clean up whitespace from column names ---
    df.columns = df.columns.str.strip()
    
    print(f"--- Statistical Summary for column '{AFFINITY_COLUMN_NAME}' ---")
    print(df[AFFINITY_COLUMN_NAME].describe())

except FileNotFoundError:
    print(f"Error: File not found at '{AFFINITY_FILE_PATH}'")
except KeyError:
    print(f"Error: Column '{AFFINITY_COLUMN_NAME}' not found in the file.")
    print("\nAvailable columns are:")
    print(df.columns.tolist()) # This will show you the actual columns found