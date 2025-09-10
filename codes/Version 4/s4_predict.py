# s4_predict.py (Definitive, Model-Aware, Fully Commented Version)
#
# PURPOSE:
# This script loads a trained "Supreme" model and a saved scaler to predict
# the binding affinity for new, unseen molecules. It is designed to be run
# after a model has been successfully trained in Stage 3.
#
# WORKFLOW:
# 1. Place FASTA files for the molecules you want to test in the 'prediction' subfolders.
# 2. If you have 3D data (.pdb), link it in the 'structure_files' section of config.json.
# 3. Update the 'prediction_parameters' in config.json to point to the correct model file.
# 4. Run the script. It requires no command-line arguments.
#
import os
import json
import numpy as np
import pandas as pd
import joblib
from Bio import SeqIO
import tensorflow as tf
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import time

# Import our definitive processors and custom objects from other project scripts
from molecule_processors import process_molecule_universal
from s3b_build_model import create_weighted_mse, PositionalEncoding

# Suppress TensorFlow and other warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore', category=UserWarning)

def load_config(config_path=None):
    """
    Loads the configuration from a JSON file.
    If no path is given, it automatically finds 'config.json' in the same directory as the script.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

def one_hot_encode_sequence(sequence, max_len):
    """Simple one-hot encoder for sequences, truncates if longer than max_len."""
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    encoded_seq = np.zeros((max_len, len(nucleotide_map)), dtype=np.float32)
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, nucleotide_map.get(char.upper(), 4)] = 1
    return encoded_seq

def prepare_input_for_prediction(primary_data, target_data, competitor_data, scaler, pad_lengths, model_inputs):
    """
    Prepares a single data point for prediction, ensuring all data matches the
    shapes required by the loaded model, including optional GNN inputs.
    """
    max_primary_len, max_target_len, max_competitor_len = pad_lengths
    
    # Prepare numerical features, ensuring the vector has the correct number of features
    num_features = [primary_data.get('gc_content', 0.5), primary_data.get('dg', 0.0), primary_data.get('conservation', 0.0)]
    if len(num_features) < scaler.n_features_in_:
        num_features += [0.0] * (scaler.n_features_in_ - len(num_features))
    scaled_numerical = scaler.transform([num_features])
    
    # Prepare sequence and 1D structure features
    primary_seq_encoded = one_hot_encode_sequence(primary_data.get('sequence', ''), max_primary_len)
    target_seq_encoded = one_hot_encode_sequence(target_data.get('sequence', ''), max_target_len)
    competitor_seq_encoded = one_hot_encode_sequence(competitor_data.get('sequence', ''), max_competitor_len)
    
    structure_vector = json.loads(primary_data.get('structure_vector', '[]'))
    structure_vector = structure_vector[:max_primary_len] # Truncate if necessary
    structure_padded = np.zeros((max_primary_len, 1), dtype=np.float32)
    structure_padded[:len(structure_vector), 0] = structure_vector
    
    inputs = {
        'primary_sequence_input': np.array([primary_seq_encoded]),
        'target_sequence_input': np.array([target_seq_encoded]),
        'competitor_sequence_input': np.array([competitor_seq_encoded]),
        'primary_structure_input': np.array([structure_padded]),
        'numerical_features_input': scaled_numerical
    }
    
    # Conditionally add GNN inputs only if the loaded model was trained with them
    if 'target_adjacency_input' in model_inputs:
        adj_matrix = np.array(json.loads(target_data.get('adjacency_matrix', '[]')))
        padded_adj = np.zeros((max_target_len, max_target_len), dtype=np.float32)
        if adj_matrix.size > 0:
            h, w = adj_matrix.shape
            padded_adj[:h, :w] = adj_matrix
        inputs['target_adjacency_input'] = np.array([padded_adj])
        
    return inputs

def load_fasta_from_folder(folder_path):
    """Loads all sequences from all FASTA files in a directory."""
    records = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return records
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.fasta', '.fa', '.fna', '.txt'))]
    for filepath in file_paths:
        records.extend(list(SeqIO.parse(filepath, "fasta")))
    return records

def main():
    total_start_time = time.time() # Overall Timer Start
    print("--- Universal Molecule Ranking Tool (with Sliding Window Support) ---")
    config = load_config()
    pred_params = config['prediction_parameters']
    train_params = config['training_parameters']
    proc_params = config.get('processing_parameters', {}) # Get processing params
    
    # --- 1. Setup paths from config ---
    project_root = config['project_root']
    experiment_id = pred_params.get('experiment_to_use')
    if not experiment_id:
        print("FATAL ERROR: 'experiment_to_use' not specified in prediction_parameters of config.json.")
        exit()

    experiment_dir = os.path.join(project_root, 'experiments', experiment_id)
    model_dir = os.path.join(experiment_dir, config['output_folders']['main_models_folder'])
    scaler_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'], 'minmax_scaler.pkl')
    prediction_dir = os.path.join(project_root, config['output_folders']['prediction_subfolder'])
    os.makedirs(prediction_dir, exist_ok=True)
    
    # --- 2. Load model and scaler ---
    model_path = os.path.join(model_dir, pred_params['model_to_use'])
    custom_objects = {}
    if train_params['advanced_training']['use_custom_loss']:
        loss_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
        custom_objects['weighted_mse'] = loss_instance
    custom_objects['PositionalEncoding'] = PositionalEncoding
        
    print("  - Loading model with custom objects...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        scaler = joblib.load(scaler_path)
        print(f"  - Successfully loaded model '{pred_params['model_to_use']}' and scaler.")
    except Exception as e:
        print(f"  - FATAL ERROR loading files: {e}")
        return

    # --- 3. Load input sequences from FASTA files ---
    input_folders = pred_params['input_folders']
    primary_records = load_fasta_from_folder(os.path.join(prediction_dir, input_folders['primary']))
    target_records = load_fasta_from_folder(os.path.join(prediction_dir, input_folders['target']))
    competitor_records = load_fasta_from_folder(os.path.join(prediction_dir, input_folders['competitor']))

    if not primary_records or not target_records:
        print("\nAborting: Missing primary sequences or target sequences.")
        return

    # --- 4. Get window size and step size from config ---
    sw_params = proc_params.get('sliding_window', {})
    use_sw = sw_params.get('use_sliding_window', False)
    window_size = sw_params.get('window_size', 500)
    step_size = sw_params.get('step_size', 250)
    print(f"\nSliding Window mode is {'ON' if use_sw else 'OFF'}. Window size: {window_size}, Step size: {step_size}")

    # --- 5. Prepare target and competitor data ---
    target_record = target_records[0]
    competitor_record = competitor_records[0] if competitor_records else None
    
    # --- NEW: SLIDING WINDOW LOGIC FOR TARGET ---
    target_seq_full = str(target_record.seq).replace('T', 'U')
    target_chunks = []
    if use_sw and len(target_seq_full) > window_size:
        for i in range(0, len(target_seq_full) - window_size + 1, step_size):
            target_chunks.append(target_seq_full[i:i+window_size])
        print(f"  - Sliced target '{target_record.id}' into {len(target_chunks)} chunks.")
    else:
        target_chunks.append(target_seq_full) # Use the whole sequence as a single chunk

    target_processed_chunks = [process_molecule_universal(((f"{target_record.id}_chunk_{i}", chunk), config, 'target_molecule')) for i, chunk in enumerate(target_chunks)]
    
    if competitor_record:
        competitor_processed = process_molecule_universal(((competitor_record.id, str(competitor_record.seq)), config, 'competitor_molecule'))
    else:
        competitor_processed = {'sequence': ''}

    # --- 6. Determine padding lengths (model expects fixed size) ---
    pad_params = train_params.get('sequence_padding', {})
    pad_lengths = (pad_params.get('max_primary_len'), pad_params.get('max_target_len'), pad_params.get('max_competitor_len'))
    print(f"  - Model expects padded lengths -> Primary: {pad_lengths[0]}, Target: {pad_lengths[1]}, Competitor: {pad_lengths[2]}")

    # --- 7. Run predictions for all primary molecules ---
    results = []
    model_input_names = list(model.input.keys())

    # ⭐ NEW: Wrap the main loop with tqdm
    for primary_record in tqdm(primary_records, desc="  Predicting affinities"):
        primary_processed = process_molecule_universal(((primary_record.id, str(primary_record.seq)), config, 'primary_molecule'))
        
        # --- NEW: Predict against each chunk and aggregate ---
        scores_with_comp = []
        scores_no_comp = []
        for target_chunk_processed in target_processed_chunks:
            inputs_with_comp = prepare_input_for_prediction(primary_processed, target_chunk_processed, competitor_processed, scaler, pad_lengths, model_input_names)
            scores_with_comp.append(model.predict(inputs_with_comp, verbose=0)[0][0])

            inputs_no_comp = prepare_input_for_prediction(primary_processed, target_chunk_processed, {'sequence': ''}, scaler, pad_lengths, model_input_names)
            scores_no_comp.append(model.predict(inputs_no_comp, verbose=0)[0][0])
        
        # Aggregate by taking the maximum affinity found across all chunks
        best_score_with_comp_transformed = max(scores_with_comp) if scores_with_comp else 0.0
        best_score_no_comp_transformed = max(scores_no_comp) if scores_no_comp else 0.0

        # --- NEW: Apply inverse transform (square) to get the real affinity score ---
        pred_with_comp = np.square(best_score_with_comp_transformed)
        pred_no_comp = np.square(best_score_no_comp_transformed)

        results.append({
            'primary_molecule_id': primary_record.id,
            'predicted_affinity_baseline': float(pred_no_comp),
            'predicted_affinity_with_competitor': float(pred_with_comp),
            'competitive_effect (higher_is_better)': float(pred_no_comp - pred_with_comp),
        })
        print(f"  - Processed {i+1}/{len(primary_records)}...", end='\r')

    print("\n\n--- Prediction Complete ---")
    total_end_time = time.time() # Overall Timer End
    print(f"Total prediction time: {total_end_time - total_start_time:.2f} seconds")
    
    
    # --- 8. Save and display results ---
    if results:
        results_df = pd.DataFrame(results).sort_values(by='competitive_effect (higher_is_better)', ascending=False)
        output_path = os.path.join(prediction_dir, pred_params['output_filename'])
        pq.write_table(pa.Table.from_pandas(results_df, preserve_index=False), output_path)
        print(f"Ranked results saved to '{output_path}'")
        print("\n--- Top 10 Candidates (Ranked by Competitive Effect) ---")
        print(results_df.head(10).to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()
    