# x3a_pre_process_conservation.py (Final, Advanced Version for Summary Counts)
import pandas as pd
import os
import re

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the 'Summary_Counts.all_predictions.txt' file you downloaded from TargetScan.
INPUT_FILE = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/conservation_score/select/Summary_Counts.all_predictions.txt"

# 2. Define where to save the clean, final conservation file.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/conservation_score/select/"
OUTPUT_FILENAME = "Conservation_Scores_Aggregated_PCT_Final.txt"
# --- END OF CONFIGURATION ---

def extract_family_from_id(mirna_id):
    """
    Extracts the core family name from a miRNA ID using regex.
    e.g., 'hsa-let-7a-1' -> 'let-7'
    e.g., 'mmu-mir-21a-5p' -> 'mir-21'
    """
    if not isinstance(mirna_id, str):
        return None
    # This regex is designed to find common miRNA family patterns like 'mir-#' or 'let-#'
    match = re.search(r"(mir-|let-)\d+[a-z]?", mirna_id.lower())
    if match:
        return match.group(0)
    return None

def preprocess_conservation_data():
    """
    Reads the large TargetScan Summary Counts file, extracts the family from the
    'Representative miRNA' column, aggregates the PCT score for each family,
    and saves a clean, lightweight text file.
    """
    print(f"--- Processing Conservation Data from: {os.path.basename(INPUT_FILE)} ---")
    
    try:
        # Define the only two columns we need to load to save memory
        columns_to_keep = ['Representative miRNA', 'Aggregate PCT']
        
        print("  - Reading the large Summary Counts file (this will take time and RAM)...")
        df = pd.read_csv(INPUT_FILE, sep='\t', usecols=columns_to_keep, na_values=["NULL"])
        
        print(f"  - Found {len(df)} total miRNA-gene interactions.")
        df.dropna(inplace=True)
        print(f"  - Kept {len(df)} interactions after removing rows with missing values.")
        
        # --- NEW: Extract the family name from the 'Representative miRNA' column ---
        print("  - Extracting miRNA family names from representative miRNAs...")
        df['miR Family extracted'] = df['Representative miRNA'].apply(extract_family_from_id)
        df.dropna(subset=['miR Family extracted'], inplace=True)

        # --- NEW: Group by the extracted family name and calculate the mean PCT score ---
        print("  - Aggregating PCT scores for each unique miRNA family...")
        family_conservation = df.groupby('miR Family extracted')['Aggregate PCT'].mean().reset_index()
        
        print(f"  - Calculated mean PCT for {len(family_conservation)} unique miRNA families.")

        # Rename columns to match what the s1a_prepare_dataset.py script expects
        family_conservation.rename(columns={
            'miR Family extracted': 'miR Family',
            'Aggregate PCT': 'PCT'
        }, inplace=True)
        
        # Save the clean, small file as a tab-separated text file
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        family_conservation.to_csv(output_path, index=False, sep='\t')
        
        print(f"\n✅ Success! Aggregated conservation file saved to:\n   {output_path}")

    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Input file not found at '{INPUT_FILE}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_conservation_data()