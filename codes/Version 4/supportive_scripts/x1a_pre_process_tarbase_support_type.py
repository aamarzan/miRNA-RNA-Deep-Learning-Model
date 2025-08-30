# x1a_pre_process_tarbase.py (Updated to generate scores from evidence)
import pandas as pd
import os
import glob

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the directory containing your raw TSV/CSV files from miRTarBase.
#    It's best to use the "strong evidence" file: miRTarBase_SE_WR.csv
INPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/other_formats/"

# 2. Define where to save the final, clean affinity file.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/"
OUTPUT_FILENAME = "miRTarBase_processed_affinity.txt"
# --- END OF CONFIGURATION ---

def assign_affinity_score(row):
    """
    Assigns a numerical score based on the 'Support Type' column.
    """
    support_type = row['Support Type']
    if support_type == 'Functional MTI':
        return 0.9  # High score for proven functional interactions
    elif support_type == 'Non-Functional MTI':
        return 0.1  # Low score for proven non-functional interactions
    else:
        return None # Return None for other types to be dropped

def preprocess_mirtarbase_directory():
    """
    Finds all miRTarBase files, extracts essential columns, creates a numerical
    affinity score based on experimental evidence, and saves a single clean file.
    """
    print(f"--- Pre-processing miRTarBase files in: {INPUT_DIR} ---")
    
    files_to_process = glob.glob(os.path.join(INPUT_DIR, "*.csv")) + glob.glob(os.path.join(INPUT_DIR, "*.tsv"))
    
    if not files_to_process:
        print("❌ ERROR: No .csv or .tsv files found in the specified input directory.")
        return

    all_dfs = []
    print("\nProcessing files...")
    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        print(f"  - Reading '{filename}'...")
        try:
            # Load the necessary columns for our logic
            columns_to_load = ['miRNA', 'Support Type']
            
            # Determine separator by file extension
            sep = '\t' if filename.lower().endswith('.tsv') else ','
            
            df = pd.read_csv(filepath, sep=sep, usecols=columns_to_load, na_values=["NA"], low_memory=False)
            
            # --- NEW: Apply our scoring logic to create the affinity score ---
            df['interaction_score'] = df.apply(assign_affinity_score, axis=1)
            
            # Drop rows that are not 'Functional MTI' or 'Non-Functional MTI'
            df.dropna(subset=['interaction_score'], inplace=True)
            
            # Keep only the final columns we need
            final_cols_df = df[['miRNA', 'interaction_score']]
            
            all_dfs.append(final_cols_df)
            print(f"    - Extracted {len(final_cols_df)} scored interactions.")
            
        except Exception as e:
            print(f"    - ⚠️ WARNING: Could not process file '{filename}'. Error: {e}")

    if not all_dfs:
        print("\n❌ ERROR: No valid data could be processed from any of the files.")
        return

    print("\n  - Combining data from all files...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # The 'miRNA' column from the file already matches our 'mirna' ID column name
    final_df.rename(columns={'miRNA': 'mirna'}, inplace=True)

    # Remove any duplicate miRNA entries, keeping the highest score
    final_df = final_df.sort_values('interaction_score', ascending=False).drop_duplicates('mirna').sort_index()
    print(f"  - Total of {len(final_df)} unique, scored interactions found.")
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✅ Success! Processed affinity file with generated scores saved to:\n   {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_mirtarbase_directory()