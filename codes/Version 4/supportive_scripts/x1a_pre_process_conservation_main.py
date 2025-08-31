# x3a_pre_process_conservation.py (Updated for new TargetScan numerical format)
import pandas as pd
import os

# --- ⚙️ USER CONFIGURATION ---
# 1. Point this to the 'miR_Family_Info.txt' file you downloaded from TargetScan.
INPUT_FILE = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/conservation_score/select/miR_Family_Info.txt"

# 2. Define where to save the clean, final conservation file.
OUTPUT_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/raw_data/conservation_score/select/"
OUTPUT_FILENAME = "Conservation_Scores_Processed.txt"
# --- END OF CONFIGURATION ---

def normalize_conservation_score(value):
    """
    Converts TargetScan's conservation integer score into a normalized 0-1 float.
    -1 (Not Conserved) -> 0.0
     0 (Poorly)        -> 0.25
     1 (Mammals)       -> 0.75
     2 (Vertebrates)   -> 1.0
    """
    score_map = {
        -1: 0.0,
        0: 0.25,
        1: 0.75,
        2: 1.0
    }
    return score_map.get(value, 0.0) # Default to 0.0 if value is unexpected

def preprocess_conservation_data():
    """
    Reads the new TargetScan family info file, selects and renames the essential
    columns, normalizes the conservation score, and saves a clean text file.
    """
    print(f"--- Pre-processing Conservation Data from: {os.path.basename(INPUT_FILE)} ---")
    
    try:
        # Define the only two columns we need to load from the new file
        columns_to_keep = ['miR family', 'Family Conservation?']
        
        print("  - Reading the TargetScan file...")
        df = pd.read_csv(INPUT_FILE, sep='\t', usecols=columns_to_keep, na_values=["NA"])
        
        print(f"  - Found {len(df)} total miRNA families.")
        df.dropna(inplace=True)
        print(f"  - Kept {len(df)} families after removing rows with missing values.")
        
        # --- NEW: Convert the integer conservation to a normalized 0-1 score ---
        df['normalized_conservation'] = df['Family Conservation?'].apply(normalize_conservation_score)
        
        # Keep only the family name and the new numerical score
        final_df = df[['miR family', 'normalized_conservation']]

        # Rename columns to match what the s1a_prepare_dataset.py script expects
        final_df.rename(columns={
            'miR family': 'miR Family',
            'normalized_conservation': 'PCT'
        }, inplace=True)
        
        # Save the clean, small file as a tab-separated text file
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        final_df.to_csv(output_path, index=False, sep='\t')
        
        print(f"\n✅ Success! Clean conservation file saved to:\n   {output_path}")

    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Input file not found at '{INPUT_FILE}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_conservation_data()