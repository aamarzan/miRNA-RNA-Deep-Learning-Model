# x2a_clean_legacy_affinity.py
import pandas as pd
import os
import glob

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the directory containing your OLD, original affinity .txt files.
INPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/legacy_files/"

# 2. Define where to save the final, clean affinity file.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/"
OUTPUT_FILENAME = "Legacy_preprocessed_affinity.txt"
# --- END OF CONFIGURATION ---

def clean_legacy_directory():
    """
    Finds all .txt files in a directory, extracts 'mirna' and 'interaction_score'
    columns, combines them, and saves a single, clean TXT file.
    """
    print(f"--- Pre-processing all legacy affinity files in: {INPUT_DIR} ---")
    
    # Find all .txt or .tsv files in the input directory
    txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    tsv_files = glob.glob(os.path.join(INPUT_DIR, "*.tsv"))
    all_files = txt_files + tsv_files
    
    if not all_files:
        print("❌ ERROR: No .txt or .tsv files found in the specified input directory.")
        return

    all_dfs = []
    print("\nProcessing files...")
    for filepath in all_files:
        filename = os.path.basename(filepath)
        print(f"  - Reading '{filename}'...")
        try:
            # Define the only two columns we need to keep
            # These names should match the columns in your old files
            columns_to_keep = ['mirna', 'interaction_score']
            
            df = pd.read_csv(filepath, sep='\t', usecols=columns_to_keep, na_values=["NA"], low_memory=False)
            df.dropna(subset=columns_to_keep, inplace=True)
            
            all_dfs.append(df)
            print(f"    - Found {len(df)} valid interactions.")
            
        except Exception as e:
            print(f"    - ⚠️ WARNING: Could not process file '{filename}'. It might be missing the required columns. Error: {e}")

    if not all_dfs:
        print("\n❌ ERROR: No valid data could be processed from any of the files.")
        return

    print("\n  - Combining data from all files...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  - Total of {len(final_df)} interactions found.")
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✅ Success! Consolidated and clean legacy affinity file saved to:\n   {output_path}")

if __name__ == "__main__":
    # Create a placeholder input directory if it doesn't exist
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created a placeholder input directory at: {INPUT_DIR}")
        print("Please place your old affinity files there before running again.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_legacy_directory()