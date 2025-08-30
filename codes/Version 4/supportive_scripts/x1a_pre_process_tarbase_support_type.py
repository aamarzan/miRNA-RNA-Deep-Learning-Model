# x1a_pre_process_tarbase.py (Updated for Multi-File and 4-Tier Scoring)
import pandas as pd
import os
import glob

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the directory containing all your raw CSV/TSV files from miRTarBase.
INPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/other_formats/"

# 2. Define where to save the final, clean affinity file.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/"
OUTPUT_FILENAME = "Final_Preprocessed_Affinity.txt"
# --- END OF CONFIGURATION ---

def assign_affinity_score(row):
    """
    Assigns a numerical score based on the 4-tier 'Support Type' criteria.
    """
    support_type = str(row.get('Support Type', ''))
    
    # Check for the strongest evidence first
    if 'Functional MTI' == support_type:
        return 0.95  # Tier 1: Strongest evidence
    elif 'Functional MTI (Weak)' in support_type:
        return 0.70  # Tier 2: Still functional, but weaker evidence
    elif 'Non-Functional MTI' == support_type:
        return 0.10  # Tier 3: Proven non-functional
    elif 'Non-Functional MTI (Weak)' in support_type:
        return 0.30  # Tier 4: Likely non-functional, weak evidence
    else:
        # Check experiments for CLIP-Seq if Support Type isn't enough
        experiments = str(row.get('Experiments', ''))
        if 'CLIP' in experiments:
             return 0.60 # Tier 2.5: Physical binding is confirmed
        return None # Return None for other types to be dropped

def preprocess_mirtarbase_directory():
    """
    Finds all miRTarBase files, extracts essential columns, creates a tiered
    numerical affinity score, handles duplicates by keeping the best evidence,
    and saves a single, consolidated clean file.
    """
    print(f"--- Pre-processing all miRTarBase files in: {INPUT_DIR} ---")
    
    files_to_process = glob.glob(os.path.join(INPUT_DIR, "*.csv")) + glob.glob(os.path.join(INPUT_DIR, "*.tsv"))
    
    if not files_to_process:
        print(f"❌ ERROR: No .csv or .tsv files found in: {INPUT_DIR}")
        return

    all_dfs = []
    print("\nProcessing files...")
    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        print(f"  - Reading '{filename}'...")
        try:
            # Load all necessary columns for our logic
            # We add 'Experiments' to help classify CLIP-Seq data
            columns_to_load = ['miRNA', 'Support Type', 'Experiments']
            sep = '\t' if filename.lower().endswith('.tsv') else ','
            
            df = pd.read_csv(filepath, sep=sep, usecols=lambda c: c in columns_to_load, na_values=["NA"], low_memory=False)
            
            # Apply our 4-tier scoring logic
            df['interaction_score'] = df.apply(assign_affinity_score, axis=1)
            df.dropna(subset=['interaction_score'], inplace=True)
            
            # Keep only the final columns we need
            final_cols_df = df[['miRNA', 'interaction_score']]
            all_dfs.append(final_cols_df)
            print(f"    - Extracted {len(final_cols_df)} scored interactions.")
            
        except Exception as e:
            print(f"    - ⚠️ WARNING: Could not process file '{filename}'. Check its format. Error: {e}")

    if not all_dfs:
        print("\n❌ ERROR: No valid data could be processed from any of the files.")
        return

    print("\n  - Combining data from all files...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Rename the miRNA column to the standard 'mirna' for the main pipeline
    final_df.rename(columns={'miRNA': 'mirna'}, inplace=True)

    print(f"  - Found {len(final_df)} total interactions before de-duplication.")

    # --- NEW: Handle duplicates by keeping the entry with the highest score ---
    final_df = final_df.sort_values('interaction_score', ascending=False).drop_duplicates('mirna').sort_index()
    print(f"  - Total of {len(final_df)} unique, high-evidence interactions remain.")
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✅ Success! Final consolidated affinity file saved to:\n   {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_mirtarbase_directory()