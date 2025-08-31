# decompress_and_rename_pdbs.py
# PURPOSE:
# A utility script to recursively find and decompress all .gz files in a
# target directory. It then renames all resulting .ent files to the
# standard .pdb extension.

import os
import glob
import gzip
import shutil

# --- ⚙️ USER CONFIGURATION ---
# 1. Set this to the absolute path of the directory you want to process.
TARGET_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/pdb_files/targets"

# 2. Set to True to automatically delete .gz files after successful decompression.
DELETE_GZ_AFTER_DECOMPRESSION = True
# --- END OF CONFIGURATION ---


def process_directory():
    """
    Main function to find, decompress .gz files, and rename .ent files.
    """
    print(f"--- Starting PDB Decompress and Rename Process ---")
    print(f"Target Directory: {TARGET_DIR}\n")

    if not os.path.isdir(TARGET_DIR):
        print(f"❌ ERROR: Target directory not found at '{TARGET_DIR}'")
        return

    # --- Step 1: Recursively Decompress All .gz Archives ---
    # This loop continues as long as it finds new .gz files to process.
    pass_num = 1
    while True:
        # Find all .gz files in the target directory
        gz_files_to_process = glob.glob(os.path.join(TARGET_DIR, "*.gz"))

        if not gz_files_to_process:
            if pass_num == 1:
                print("  - No .gz files found to decompress.")
            else:
                print("  - No more .gz files found to process.")
            break # Exit the loop if there are no zips left

        print(f"--- Pass {pass_num}: Found {len(gz_files_to_process)} .gz file(s) ---")

        for gz_path in gz_files_to_process:
            filename = os.path.basename(gz_path)
            # Determine the output path by removing the .gz extension
            output_path = os.path.splitext(gz_path)[0]
            
            print(f"  - Decompressing: '{filename}' -> '{os.path.basename(output_path)}'")
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Optionally, delete the .gz file after processing
                if DELETE_GZ_AFTER_DECOMPRESSION:
                    os.remove(gz_path)
                    print(f"    - Deleted archive: '{filename}'")

            except Exception as e:
                print(f"    - ❌ ERROR processing '{filename}': {e}")
        pass_num += 1
    
    # --- Step 2: Rename all .ent files to .pdb ---
    print("\n--- Renaming .ent files to .pdb ---")
    ent_files = glob.glob(os.path.join(TARGET_DIR, "*.ent"))

    if not ent_files:
        print("  - No .ent files found to rename.")
    else:
        renamed_count = 0
        for ent_path in ent_files:
            try:
                base_name = os.path.splitext(ent_path)[0]
                pdb_path = base_name + ".pdb"
                
                # Check if a file with the .pdb name already exists to avoid errors
                if os.path.exists(pdb_path):
                    print(f"  - ⚠️ WARNING: '{os.path.basename(pdb_path)}' already exists. Skipping rename for '{os.path.basename(ent_path)}'.")
                    continue
                    
                os.rename(ent_path, pdb_path)
                renamed_count += 1
            except Exception as e:
                print(f"  - ❌ ERROR renaming '{os.path.basename(ent_path)}': {e}")
        print(f"  - Successfully renamed {renamed_count} file(s).")
        
    print("\n✅ --- Process Complete ---")


if __name__ == "__main__":
    process_directory()