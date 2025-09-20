# codes/check_parquet_columns.py
import pyarrow.parquet as pq
import os
import json

# --- Simple Config Loader to find the project root ---
def get_project_root():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get('project_root')
    except FileNotFoundError:
        return '.' # Default to current directory

# --- Main script ---
def main():
    project_root = get_project_root()
    prepared_folder = os.path.join(project_root, 'dataset', 'prepared_dataset')
    
    try:
        # Find the newest .parquet file in the folder
        parquet_files = [f for f in os.listdir(prepared_folder) if f.endswith('.parquet') and os.path.isfile(os.path.join(prepared_folder, f))]
        if not parquet_files:
            print(f"No Parquet files found in '{prepared_folder}'")
            return
            
        latest_file = sorted(parquet_files)[-1]
        file_path = os.path.join(prepared_folder, latest_file)
        
        print(f"--- Checking columns for file: {file_path} ---")
        
        # Read the schema and print the column names
        schema = pq.read_schema(file_path)
        print("\nFound the following columns:")
        for name in schema.names:
            print(f"- {name}")

        # Check for the critical missing column
        if 'primary_sequence' not in schema.names:
            print("\n*** CRITICAL ERROR: The 'primary_sequence' column is MISSING! ***")
            print("This is the root cause of your problem in Stage 3.")
        else:
            print("\nSuccess! All critical columns appear to be present.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()