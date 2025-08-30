# consolidate_affinity.py
# PURPOSE:
# To read all raw affinity files from a directory, intelligently find the
# miRNA ID and score columns, consolidate the data, resolve duplicates by
# keeping the highest score for each unique miRNA, and save a single,
# clean master affinity file for the main pipeline.
#
import pandas as pd
import os
import glob

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the directory containing ALL your raw affinity files
#    (e.g., from TarBase, your legacy files, etc.).
INPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/"

# 2. Define the name for the final, consolidated output file.
OUTPUT_FILENAME = "Master_Affinity_Scores.txt"
# --- END OF CONFIGURATION ---

def find_column_names(df_columns):
    """
    Intelligently finds the correct miRNA ID and score column names from a list
    of potential candidates.
    """
    # Define standard names and known alternatives
    id_col_target = 'mirna'
    id_alternatives = ['miRNA', 'mirna_name', 'mirna_id']
    
    score_col_target = 'interaction_score'
    score_alternatives = ['microt_score', 'Score', 'Affinity']
    
    found_id_col = None
    found_score_col = None

    # Find the ID column
    if id_col_target in df_columns:
        found_id_col = id_col_target
    else:
        for alt in id_alternatives:
            if alt in df_columns:
                found_id_col = alt
                break
    
    # Find the Score column
    if score_col_target in df_columns:
        found_score_col = score_col_target
    else:
        for alt in score_alternatives:
            if alt in df_columns:
                found_score_col = alt
                break
                
    if not found_id_col or not found_score_col:
        raise ValueError(f"Could not find the required ID or score columns in a file. Found columns: {list(df_columns)}")
        
    return found_id_col, found_score_col

def consolidate_affinity_files():
    """
    Main function to process the affinity data directory.
    """
    print(f"--- Consolidating all affinity files in: {INPUT_DIR} ---")
    
    files_to_process = glob.glob(os.path.join(INPUT_DIR, "*.csv")) + \
                       glob.glob(os.path.join(INPUT_DIR, "*.tsv")) + \
                       glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    
    # Exclude the potential output file from being read as an input
    output_path = os.path.join(INPUT_DIR, OUTPUT_FILENAME)
    files_to_process = [f for f in files_to_process if f != output_path]

    if not files_to_process:
        print(f"❌ ERROR: No .csv, .tsv, or .txt files found in: {INPUT_DIR}")
        return

    all_dfs = []
    print("\nProcessing files...")
    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        print(f"  - Reading '{filename}'...")
        try:
            sep = '\t' if filename.lower().endswith(('.tsv', '.txt')) else ','
            df = pd.read_csv(filepath, sep=sep, na_values=["NA"], low_memory=False)
            
            # Intelligently find the correct column names
            id_col, score_col = find_column_names(df.columns)
            
            # Select and rename to our standard format
            df = df[[id_col, score_col]].rename(columns={
                id_col: 'mirna',
                score_col: 'interaction_score'
            })

            df.dropna(inplace=True)
            all_dfs.append(df)
            print(f"    - Found {len(df)} valid interactions.")
            
        except Exception as e:
            print(f"    - ⚠️ WARNING: Could not process file '{filename}'. Check its format. Error: {e}")

    if not all_dfs:
        print("\n❌ ERROR: No valid data could be processed from any of the files.")
        return

    print("\n  - Combining data from all files...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"  - Found {len(final_df)} total interactions before de-duplication.")

    # De-duplicate by keeping only the highest score for each unique miRNA
    final_df = final_df.sort_values('interaction_score', ascending=False).drop_duplicates('mirna').sort_index()
    print(f"  - Total of {len(final_df)} unique, high-evidence interactions remain.")
    
    # Save the final consolidated file
    final_df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✅ Success! Master affinity file saved to:\n   {output_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created a placeholder input directory at: {INPUT_DIR}")
        print("Please place your raw affinity files there before running again.")
    consolidate_affinity_files()