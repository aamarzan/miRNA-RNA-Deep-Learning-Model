# unzip_and_rename_pdbs.py
# PURPOSE:
# A utility script to recursively find and process all .zip archives in a
# target directory. It extracts only the PDB files (with .ent extension)
# and then renames all extracted .ent files to the standard .pdb extension.
# It also handles zip files found inside other zip files.

import os
import glob
import zipfile

# --- ⚙️ USER CONFIGURATION ---
# 1. Set this to the absolute path of the directory you want to process.
#    This can be your 'targets' or 'competitors' folder.
TARGET_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/pdb_files/targets"

# 2. Set to True to automatically delete .zip files after successful extraction.
DELETE_ZIPS_AFTER_EXTRACTION = False
# --- END OF CONFIGURATION ---


def process_directory():
    """
    Main function to find, extract, and rename PDB files.
    """
    print(f"--- Starting PDB Unzip and Rename Process ---")
    print(f"Target Directory: {TARGET_DIR}\n")

    if not os.path.isdir(TARGET_DIR):
        print(f"❌ ERROR: Target directory not found at '{TARGET_DIR}'")
        return

    # --- Step 1: Recursively Unzip All Archives ---
    # This loop continues as long as it finds new .zip files to process,
    # which handles the case of zip files inside other zip files.
    while True:
        # Find all .zip files in the target directory
        zip_files_to_process = glob.glob(os.path.join(TARGET_DIR, "*.zip"))

        if not zip_files_to_process:
            print("  - No more .zip files found to process.")
            break # Exit the loop if there are no zips left

        print(f"Found {len(zip_files_to_process)} zip file(s) in this pass...")

        for zip_path in zip_files_to_process:
            filename = os.path.basename(zip_path)
            print(f"  - Processing archive: '{filename}'")
            try:
                with zipfile.ZipFile(zip_path, 'r') as archive:
                    # Find all members that are .ent files or nested .zip files
                    members_to_extract = [
                        member for member in archive.namelist()
                        if member.lower().endswith('.ent') or member.lower().endswith('.zip')
                    ]

                    if members_to_extract:
                        print(f"    - Found {len(members_to_extract)} relevant file(s) to extract.")
                        archive.extractall(path=TARGET_DIR, members=members_to_extract)
                    else:
                        print(f"    - No .ent or nested .zip files found in this archive.")

                # Optionally, delete the zip file after processing
                if DELETE_ZIPS_AFTER_EXTRACTION:
                    os.remove(zip_path)
                    print(f"    - Deleted archive: '{filename}'")

            except zipfile.BadZipFile:
                print(f"    - ⚠️ WARNING: Skipping corrupted or invalid zip file: '{filename}'")
            except Exception as e:
                print(f"    - ❌ ERROR processing '{filename}': {e}")
    
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
                os.rename(ent_path, pdb_path)
                renamed_count += 1
            except Exception as e:
                print(f"  - ❌ ERROR renaming '{os.path.basename(ent_path)}': {e}")
        print(f"  - Successfully renamed {renamed_count} file(s).")
        
    print("\n✅ --- Process Complete ---")


if __name__ == "__main__":
    process_directory()