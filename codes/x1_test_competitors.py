# codes/x1_test_competitors.py
import os
import json
from molecule_processors import process_molecule_universal, load_config
from s1_prepare_dataset import load_data_from_fasta # Re-use the loader

def run_diagnostic():
    """Tests each competitor sequence and provides a detailed success/failure report."""
    print("--- Starting Competitor Sequence Diagnostic Test ---")
    
    config = load_config()
    if not config:
        print("FATAL: Could not load config.json")
        return

    # --- Load Competitor Data ---
    params = config.get('processing_parameters', {})
    competitor_type = config['experiment_setup']['competitor_molecule']
    role_key = f"{competitor_type.lower()}_competitor"
    
    competitor_folder = config['data_sources'][role_key]['folder']
    data_path = os.path.join(config['project_root'], 'dataset', competitor_folder, 'select')
    
    print(f"\nLoading competitor FASTA files from: {data_path}\n")
    competitor_sequences = load_data_from_fasta(data_path)

    if not competitor_sequences:
        print("No competitor sequences found to test.")
        return

    # --- Test Each Sequence Individually ---
    success_count = 0
    failure_count = 0
    for mol_id, sequence in competitor_sequences.items():
        print(f"Testing ID: {mol_id}...")
        try:
            # We are testing the universal processor directly
            args = ((mol_id, sequence), params, role_key)
            result = process_molecule_universal(args)

            if isinstance(result, dict) and 'sequence' in result:
                print(f"  -> SUCCESS: Processed successfully.\n")
                success_count += 1
            else:
                # The processor returned a rejection tuple, e.g., (id, 'reject_reason')
                reason = result[1] if isinstance(result, tuple) else "Unknown"
                print(f"  -> FAILURE: Molecule was rejected. Reason: {reason}\n")
                failure_count += 1
        except Exception as e:
            # The processor raised an unexpected error
            print(f"  -> CRITICAL FAILURE: An unexpected error occurred: {e}\n")
            failure_count += 1
            
    print("--- Diagnostic Complete ---")
    print(f"Total Sequences Tested: {len(competitor_sequences)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {failure_count}")


if __name__ == "__main__":
    run_diagnostic()