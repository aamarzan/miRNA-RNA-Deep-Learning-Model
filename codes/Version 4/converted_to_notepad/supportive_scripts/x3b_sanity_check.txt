# codes/3_sanity_check.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# This is a simplified DataGenerator for our test
class SanityCheckGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, key_to_test):
        self.data_path = data_path
        self.key_to_test = key_to_test
        
        # We will try to load ONLY the problematic file and the target file
        train_input_path = os.path.join(data_path, f'X_train_{key_to_test}.npy')
        train_target_path = os.path.join(data_path, 'y_train.npy')
        
        print(f"--- Sanity Check ---")
        print(f"Attempting to load input: {train_input_path}")
        print(f"Attempting to load target: {train_target_path}")
        
        if not os.path.exists(train_input_path):
            raise FileNotFoundError(f"CRITICAL FAILURE: The input file does not exist: {train_input_path}")
        
        self.input_data = np.load(train_input_path, mmap_mode='r')
        self.target_data = np.load(train_target_path, mmap_mode='r')
        
        print(f"SUCCESS: Successfully loaded both files.")
        print(f"Input data shape: {self.input_data.shape}")
        
    def __len__(self):
        # Just run for one batch
        return 1

    def __getitem__(self, index):
        # Return the first 32 samples
        X = {self.key_to_test: self.input_data[:32]}
        y = self.target_data[:32]
        return X, y

# --- Main Execution Block ---
def main():
    # Find the project root to locate the config file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_path = os.path.join(config['project_root'], config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    
    key_we_are_testing = 'primary_sequence_input'
    
    print(f"--- Starting Sanity Check for key: '{key_we_are_testing}' ---")
    
    try:
        # 1. Test the Data Loading
        generator = SanityCheckGenerator(data_path, key_we_are_testing)
        
        # 2. Test a Minimal Model
        print("\n--- Building Minimal Model ---")
        sample_X, _ = generator[0]
        input_shape = sample_X[key_we_are_testing].shape[1:]
        
        input_layer = Input(shape=input_shape, name=key_we_are_testing)
        x = Flatten()(input_layer)
        output_layer = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer='adam')
        
        print("SUCCESS: Minimal model built and compiled without errors.")
        print("\n--- SANITY CHECK PASSED ---")

    except Exception as e:
        print(f"\n--- SANITY CHECK FAILED ---")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()