# process_pdb_archives.py
# PURPOSE:
# A universal utility to handle the multi-layer archives from RCSB PDB.
# 1. Unzips all top-level .zip files.
# 2. Scans all subdirectories for .gz files.
# 3. Decompresses .pdb.gz and .ent.gz files.
# 4. Moves the final .pdb/.ent files to the main target directory.
# 5. Renames any .ent files to .pdb.
# 6. Cleans up the original archives and empty subfolders.
#
import os
import glob
import zipfile
import gzip
import shutil

# --- ⚙️ USER CONFIGURATION ---
# 1. Set this to the absolute path of the directory you want to process.
TARGET_DIR = "E:/1. miRNA-RNA-Deep-Learning-Model/dataset/pdb_files/targets"

# 2. Set to True to automatically delete original .zip/.gz files and subfolders.
CLEANUP_AFTER_PROCESSING = True
# --- END OF CONFIGURATION ---


def process_pdb_archives():
    """
    Main function to orchestrate the multi-layer extraction and renaming process.
    """
    print(f"--- Starting Universal PDB Archive Processor ---")
    print(f"Target Directory: {TARGET_DIR}\n")

    if not os.path.isdir(TARGET_DIR):
        print(f"❌ ERROR: Target directory not found at '{TARGET_DIR}'")
        return

    # --- Step 1: Unzip all top-level .zip archives ---
    top_level_zips = glob.glob(os.path.join(TARGET_DIR, "*.zip"))
    if top_level_zips:
        print(f"--- Found {len(top_level_zips)} top-level .zip archives to extract ---")
        for zip_path in top_level_zips:
            filename = os.path.basename(zip_path)
            print(f"  - Unzipping '{filename}'...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as archive:
                    archive.extractall(path=TARGET_DIR)
                if CLEANUP_AFTER_PROCESSING:
                    os.remove(zip_path)
                    print(f"    - Deleted archive: '{filename}'")
            except Exception as e:
                print(f"    - ❌ ERROR unzipping '{filename}': {e}")
    else:
        print("--- No top-level .zip archives found ---")

    # --- Step 2: Find and decompress all nested .gz files ---
    print("\n--- Searching for and decompressing all .pdb.gz/.ent.gz files ---")
    gz_files = glob.glob(os.path.join(TARGET_DIR, "**", "*.gz"), recursive=True)
    
    decompressed_files = 0
    if gz_files:
        for gz_path in gz_files:
            filename = os.path.basename(gz_path)
            # Determine the final output path in the main TARGET_DIR
            final_name = os.path.basename(os.path.splitext(gz_path)[0]) # e.g., '3dox.pdb'
            final_path = os.path.join(TARGET_DIR, final_name)
            
            print(f"  - Decompressing '{filename}' -> '{final_name}'")
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(final_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                decompressed_files += 1
                if CLEANUP_AFTER_PROCESSING:
                    os.remove(gz_path) # Delete the .gz file
            except Exception as e:
                print(f"    - ❌ ERROR decompressing '{filename}': {e}")
        print(f"  - Decompressed {decompressed_files} files.")
    else:
        print("  - No .gz files found in any subdirectory.")

    # --- Step 3: Rename any .ent files to .pdb ---
    print("\n--- Renaming any remaining .ent files to .pdb ---")
    ent_files = glob.glob(os.path.join(TARGET_DIR, "*.ent"))
    if ent_files:
        renamed_count = 0
        for ent_path in ent_files:
            base_name = os.path.splitext(ent_path)[0]
            pdb_path = base_name + ".pdb"
            if not os.path.exists(pdb_path):
                os.rename(ent_path, pdb_path)
                renamed_count += 1
        print(f"  - Renamed {renamed_count} file(s).")
    else:
        print("  - No .ent files found to rename.")

    # --- Step 4: Clean up empty subfolders ---
    if CLEANUP_AFTER_PROCESSING:
        print("\n--- Cleaning up empty subfolders ---")
        cleaned_count = 0
        # Walk directory from bottom up to safely delete empty folders
        for dirpath, dirnames, filenames in os.walk(TARGET_DIR, topdown=False):
            if not dirnames and not filenames and dirpath != TARGET_DIR:
                try:
                    os.rmdir(dirpath)
                    print(f"  - Removed empty folder: {os.path.basename(dirpath)}")
                    cleaned_count += 1
                except OSError as e:
                    print(f"  - ❌ ERROR removing folder '{dirpath}': {e}")
        if cleaned_count == 0:
            print("  - No empty folders to clean.")

    print("\n✅ --- Process Complete ---")

if __name__ == "__main__":
    process_pdb_archives()