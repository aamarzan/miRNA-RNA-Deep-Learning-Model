# x1a_pre_process_tarbase.py (Updated for Multi-Level Tiered Scoring)
import pandas as pd
import os
import glob

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the directory containing all your raw CSV/TSV files from miRTarBase.
INPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/other_formats/"

# 2. Define where to save the final, clean affinity file.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/affinity_score/select/"
OUTPUT_FILENAME = "Multi-Level_preprocessed_affinity.txt"
# --- END OF CONFIGURATION ---

def assign_affinity_score(row, evidence_level):
    """
    Assigns a numerical score based on the 'Support Type' column and the
    evidence level of the source file ('strong' or 'weak').
    """
    support_type = row.get('Support Type', '')
    if 'Functional MTI' in support_type:
        return 0.95 if evidence_level == 'strong' else 0.65
    elif 'Non-Functional MTI' in support_type:
        return 0.1
    else:
        return None # Return None for other types to be dropped

def preprocess_mirtarbase_directory():
    """
    Finds all miRTarBase files, extracts essential columns, creates a multi-level
    affinity score, and saves a single, consolidated clean file.
    """
    print(f"--- Pre-processing all miRTarBase files with Multi-Level Scoring in: {INPUT_DIR} ---")
    
    files_to_process = glob.glob(os.path.join(INPUT_DIR, "*.csv")) + glob.glob(os.path.join(INPUT_DIR, "*.tsv"))
    
    if not files_to_process:
        print(f"❌ ERROR: No .csv or .tsv files found in the specified input directory: {INPUT_DIR}")
        return

    all_dfs = []
    print("\nProcessing files...")
    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        print(f"  - Reading '{filename}'...")
        try:
            # Determine evidence level from filename
            if "_SE_" in filename:
                evidence_level = 'strong'
            else: # Treat general files like MTI.csv and _WE_ files as 'weak' evidence
                evidence_level = 'weak'
            print(f"    - Detected evidence level: '{evidence_level}'")

            columns_to_load = ['miRNA', 'Support Type']
            sep = '\t' if filename.lower().endswith('.tsv') else ','
            
            df = pd.read_csv(filepath, sep=sep, usecols=columns_to_load, na_values=["NA"], low_memory=False)
            
            # Apply our scoring logic to create the affinity score
            df['interaction_score'] = df.apply(assign_affinity_score, axis=1, evidence_level=evidence_level)
            
            df.dropna(subset=['interaction_score'], inplace=True)
            
            final_cols_df = df[['miRNA', 'interaction_score']]
            all_dfs.append(final_cols_df)
            print(f"    - Extracted {len(final_cols_df)} scored interactions.")
            
        except Exception as e:
            print(f"    - ⚠️ WARNING: Could not process file '{filename}'. It might be missing required columns. Error: {e}")

    if not all_dfs:
        print("\n❌ ERROR: No valid data could be processed from any of the files.")
        return

    print("\n  - Combining data from all files...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    final_df.rename(columns={'miRNA': 'mirna'}, inplace=True)

    # Remove duplicates, always keeping the interaction with the HIGHEST score/evidence
    final_df = final_df.sort_values('interaction_score', ascending=False).drop_duplicates('mirna').sort_index()
    print(f"  - Total of {len(final_df)} unique, scored interactions found after de-duplication.")
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n✅ Success! Multi-level affinity file saved to:\n   {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_mirtarbase_directory()