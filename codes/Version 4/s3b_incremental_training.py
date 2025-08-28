# s3b_incremental_training.py (Fully Config-Driven, Final Version)
import os
import json
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, TensorBoard)

# <<< FIX: Import from the correctly named s3_build_model.py script >>>
from s3a_build_model import DataGenerator, create_weighted_mse

# --- Configuration Loader ---
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

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Step 1: Load Config and Define Paths ---
    config = load_config()
    inc_params = config['incremental_training']
    train_params = config['training_parameters']
    
    project_root = config['project_root']
    model_save_dir = os.path.join(project_root, config['output_folders']['main_models_folder'])
    logs_dir = os.path.join(project_root, config['output_folders']['logs_subfolder'])
    
    new_data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], inc_params['new_data_subfolder'])
    existing_model_path = os.path.join(model_save_dir, inc_params['existing_model_name'])

    # --- Step 2: Load the PREVIOUSLY trained model ---
    print(f"--- Starting Incremental Training ---")
    print(f"Loading existing model from: {existing_model_path}")
    
    custom_objects = {}
    if train_params['advanced_training']['use_custom_loss']:
        loss_func_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
        custom_objects = {'weighted_mse': loss_func_instance}
        print("  - Custom loss function 'weighted_mse' will be used for loading.")

    try:
        model = load_model(existing_model_path, custom_objects=custom_objects)
        print("  - Model loaded successfully.")
    except Exception as e:
        print(f"  - FATAL ERROR loading model: {e}")
        exit()

    # --- Step 3: Prepare the NEW data generators ---
    print(f"\nPreparing new data from: {new_data_path}")
    try:
        train_indices = np.arange(len(np.load(os.path.join(new_data_path, 'y_train.npy'))))
        test_indices = np.arange(len(np.load(os.path.join(new_data_path, 'y_test.npy'))))
        np.random.shuffle(train_indices)
    except FileNotFoundError as e:
        print(f"  - Error finding new data files: {e}. Please run data prep for the new dataset first.")
        exit()
    
    # --- FIX: Pass the model_inputs from config ---
    model_inputs = train_params.get('model_inputs')
    if not model_inputs:
        print("FATAL: 'model_inputs' key not found in training_parameters of config.json.")
        exit()
        
    new_train_generator = DataGenerator(new_data_path, train_params['batch_size'], train_indices, 'X_train_', train_params['advanced_training'], model_inputs)
    new_test_generator = DataGenerator(new_data_path, train_params['batch_size'], test_indices, 'X_test_', train_params['advanced_training'], model_inputs)

    # --- Step 4: Re-compile the model with a VERY LOW learning rate ---
    print("\nRe-compiling model with a low learning rate for fine-tuning...")
    low_learning_rate = inc_params['fine_tune_learning_rate']
    loss_function = custom_objects.get('weighted_mse', 'mean_squared_error')
    
    model.compile(optimizer=Adam(learning_rate=low_learning_rate), 
                  loss=loss_function, 
                  metrics=['mean_absolute_error'])
    model.summary()

    # --- Step 5: Set up new callbacks ---
    new_model_filepath = os.path.join(model_save_dir, inc_params['new_model_name'])
    log_dir = os.path.join(logs_dir, "fit", f"incremental_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    callbacks = [
        ModelCheckpoint(filepath=new_model_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=train_params['advanced_training'].get('early_stopping_patience', 10), mode='min', restore_best_weights=True),
        TensorBoard(log_dir=log_dir)
    ]

    # --- Step 6: Continue training on the NEW data ---
    print("\nStarting fine-tuning on the new dataset...")
    history = model.fit(
        new_train_generator,
        epochs=inc_params['fine_tune_epochs'],
        validation_data=new_test_generator,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n--- Incremental Training Complete ---")
    print(f"The newly fine-tuned model has been saved to: {new_model_filepath}")