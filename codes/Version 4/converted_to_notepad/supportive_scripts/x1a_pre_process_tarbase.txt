# x1a_pre_process_tarbase.py (Updated to use mirna_name)
import pandas as pd
import os
import glob

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the directory containing your raw TSV files from TarBase.
INPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/other_formats/"

# 2. Define where to save the final, clean affinity file.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/"
OUTPUT_FILENAME = "TarBase_preprocessed_affinity.txt"
# --- END OF CONFIGURATION ---

def preprocess_tarbase_directory():
    """
    Finds all .tsv files, extracts 'mirna_name' and 'microt_score',
    renames them, combines them, and saves a single, clean TXT file.
    """
    print(f"--- Pre-processing all TarBase files in: {INPUT_DIR} ---")
    
    tsv_files = glob.glob(os.path.join(INPUT_DIR, "*.tsv"))
    
    if not tsv_files:
        print("❌ ERROR: No .tsv files found in the specified input directory.")
        return

    all_dfs = []
    print("\nProcessing files...")
    for filepath in tsv_files:
        filename = os.path.basename(filepath)
        print(f"  - Reading '{filename}'...")
        try:
            # --- FIX: Changed 'mirna_id' to 'mirna_name' ---
            columns_to_keep = ['mirna_name', 'microt_score']
            
            df = pd.read_csv(filepath, sep='\t', usecols=columns_to_keep, na_values=["NA"], low_memory=False)
            df.dropna(subset=columns_to_keep, inplace=True)
            
            all_dfs.append(df)
            print(f"    - Found {len(df)} valid interactions.")
            
        except Exception as e:
            print(f"    - ⚠️ WARNING: Could not process file '{filename}'. Error: {e}")

    if not all_dfs:
        print("\n❌ ERROR: No valid data could be processed from any of the files.")
        return

    print("\n  - Combining data from all files...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  - Total of {len(final_df)} interactions found.")

    # --- FIX: Changed 'mirna_id' to 'mirna_name' for renaming ---
    final_df.rename(columns={
        'mirna_name': 'mirna',
        'microt_score': 'interaction_score'
    }, inplace=True)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    # --- FIX: Save as a tab-separated .txt file ---
    final_df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✅ Success! Consolidated and clean affinity file saved to:\n   {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_tarbase_directory()