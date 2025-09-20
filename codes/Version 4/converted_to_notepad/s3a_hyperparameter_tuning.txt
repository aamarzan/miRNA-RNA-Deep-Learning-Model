# s3a_hyperparameter_tuning.py
# PURPOSE:
# To automatically find the best hyperparameters for the "Supreme" model.
# This script uses KerasTuner to systematically test different combinations
# of network sizes, dropout rates, and learning rates to find the optimal
# configuration for your specific dataset.
#
import os
import json
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout, BatchNormalization,
                                     concatenate, Bidirectional, LSTM, Layer,
                                     MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import time

# --- FIX: Import from the correctly renamed s3b_build_model.py script ---
from s3b_build_model import DataGenerator, PositionalEncoding
from spektral.layers import GCSConv

def load_config(config_path=None):
    """Loads the configuration from the JSON file in the script's directory."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def build_hyper_model(hp):
    """
    This function builds the model but defines a search space for key hyperparameters.
    KerasTuner will call this function repeatedly to build and test different model versions.
    """
    config = load_config()
    params = config['training_parameters']
    
    # --- Dynamically define input shapes from config.json ---
    sequence_padding = params['sequence_padding']
    input_shapes = {
        'primary_sequence_input': (sequence_padding['max_primary_len'], 5),
        'target_sequence_input': (sequence_padding['max_target_len'], 5),
        'competitor_sequence_input': (sequence_padding['max_competitor_len'], 5),
        'primary_structure_input': (sequence_padding['max_primary_len'], 1),
        'numerical_features_input': (len(params['numerical_features']),),
        'target_adjacency_input': (sequence_padding['max_target_len'], sequence_padding['max_target_len'])
    }
    
    input_layers = {key: Input(shape=shape, name=key) for key, shape in input_shapes.items() if key in params['model_inputs']}

    # --- Define Hyperparameter Search Space ---
    hp_cnn_filters = hp.Choice('cnn_filters', values=[32, 64, 128])
    hp_lstm_units = hp.Choice('lstm_units', values=[32, 64])
    hp_dense_units = hp.Choice('dense_units', values=[128, 256])
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    # --- Build Model using Hyperparameters ---
    def create_seq_processor(input_tensor, cnn_filters, lstm_units):
        max_len, features = input_tensor.shape[1], input_tensor.shape[2]
        pos_encoded_input = PositionalEncoding(max_len, features)(input_tensor)
        x = Conv1D(filters=cnn_filters, kernel_size=7, padding='same', activation='relu')(pos_encoded_input)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        return x

    primary_seq_processed = create_seq_processor(input_layers['primary_sequence_input'], hp_cnn_filters, hp_lstm_units)
    target_seq_processed = create_seq_processor(input_layers['target_sequence_input'], hp_cnn_filters, hp_lstm_units)
    competitor_seq_processed = create_seq_processor(input_layers['competitor_sequence_input'], hp_cnn_filters, hp_lstm_units)
    
    attention_output = MultiHeadAttention(num_heads=4, key_dim=hp_lstm_units)(query=primary_seq_processed, value=target_seq_processed, key=target_seq_processed)
    attention_output = LayerNormalization()(attention_output + primary_seq_processed)
    
    features_to_combine = [
        GlobalAveragePooling1D()(attention_output),
        GlobalAveragePooling1D()(target_seq_processed),
        GlobalAveragePooling1D()(competitor_seq_processed),
        GlobalAveragePooling1D()(Conv1D(32, 5, activation='relu')(input_layers['primary_structure_input'])),
        Dense(16, activation='relu')(input_layers['numerical_features_input'])
    ]
    
    combined = concatenate(features_to_combine)
    combined = Dropout(hp_dropout_rate)(combined)
    
    x = Dense(hp_dense_units, activation='relu', kernel_regularizer=l2(0.001))(combined)
    x = BatchNormalization()(x)
    x = Dropout(hp_dropout_rate)(x)
    x = Dense(hp_dense_units // 2, activation='relu', kernel_regularizer=l2(0.001))(x)
    output = Dense(1, activation='sigmoid', name='affinity_output')(x)

    model = Model(inputs=list(input_layers.values()), outputs=output)
    
    # NOTE: Using a standard loss for tuning is often more stable.
    # The custom weighted loss can be used in the final training run.
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

if __name__ == "__main__":
    config = load_config()
    params = config['training_parameters']
    
    # --- Load Data ---
    project_root = config['project_root']
    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    
    train_indices = np.arange(len(np.load(os.path.join(data_path, 'y_train.npz'))['data']))
    test_indices = np.arange(len(np.load(os.path.join(data_path, 'y_test.npz'))['data']))
    np.random.shuffle(train_indices)

    adv_params = params['advanced_training']
    model_inputs = params.get('model_inputs')

    train_generator = DataGenerator(data_path, params['batch_size'], train_indices, 'X_train_', adv_params, model_inputs)
    val_generator = DataGenerator(data_path, params['batch_size'], test_indices, 'X_test_', adv_params, model_inputs)

    # --- Setup and Run Tuner ---
    tuner = kt.Hyperband(
        build_hyper_model,
        objective='val_mean_absolute_error',
        max_epochs=40, # Max epochs to train a model for
        factor=3,
        directory=os.path.join(project_root, 'experiments', 'kerastuner'),
        project_name=config.get('experiment_id', 'hyperparam_search')
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    print("\n--- Starting Hyperparameter Search ---")
    start_time = time.time() # Timer Start
    tuner.search(train_generator, epochs=params['epochs'], validation_data=val_generator, callbacks=[stop_early])
    end_time = time.time() # Timer End
    
    # --- Get and Print Best Hyperparameters ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    --- Hyperparameter Search Complete ---
    The optimal number of CNN filters is {best_hps.get('cnn_filters')}.
    The optimal number of LSTM units is {best_hps.get('lstm_units')}.
    The optimal number of Dense units is {best_hps.get('dense_units')}.
    The optimal dropout rate is {best_hps.get('dropout_rate')}.
    The optimal learning rate for the Adam optimizer is {best_hps.get('learning_rate')}.
    --------------------------------------
    You can now update these values in your config.json and run the main training 
    script (s3b_build_model.py) for a final, fully trained model.
    """)
    print(f"Total time taken for search: {end_time - start_time:.2f} seconds")