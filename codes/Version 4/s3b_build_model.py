# s3_build_model.py (Definitive, Fully Commented Version)
#
# PURPOSE:
# This is the primary model training script. It loads the final processed data
# (in compressed .npz format), builds the "Supreme" hybrid model architecture
# (CNN-LSTM-Attention-GNN), and runs the training process. It is fully
# controlled by the parameters set in the 'config.json' file.
#
# WORKFLOW:
# This script is the main event (Stage 3). It should be run after the full
# data preparation pipeline (s1, s1b, s2, s2b) is complete and the final
# .npz files are ready in the 'processed_for_dl' folder.
#
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout, BatchNormalization,
                                     concatenate, Bidirectional, LSTM, Layer,
                                     MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.regularizers import l2
import datetime
import time
from spektral.layers import GCSConv
import matplotlib
matplotlib.use('Agg')  # Safe for headless servers
import matplotlib.pyplot as plt

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

# --- Custom Data Generator ---
class DataGenerator(tf.keras.utils.Sequence):
    """
    A memory-safe data generator that loads data in batches from disk.
    It dynamically discovers which of the configured input files are available.
    """
    def __init__(self, data_path, batch_size, indices, prefix, advanced_params, model_inputs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.indices = indices
        self.prefix = prefix
        self.use_sample_weights = advanced_params.get('use_sample_weighting', False)
        self.weight_alpha = advanced_params.get('sample_weight_alpha', 10.0)
        
        # --- MODIFICATION: Reads input keys from config ---
        potential_keys = model_inputs
        
        # Scan the directory and find which of the potential files actually exist
        self.input_keys = []
        print(f"  - DataGenerator searching for configured inputs in: {data_path}")
        for key in potential_keys:
            filepath = os.path.join(self.data_path, f'{self.prefix}{key}.npz')
            if os.path.exists(filepath):
                self.input_keys.append(key)
        
        if not self.input_keys:
            raise FileNotFoundError(f"No input .npz files matching configured keys found in '{data_path}'.")
        print(f"  - Found {len(self.input_keys)} available input sources: {sorted(self.input_keys)}")

        self.inputs = {}
        for key in self.input_keys:
            npz_file_path = os.path.join(data_path, f'{self.prefix}{key}.npz')
            with np.load(npz_file_path, mmap_mode='r') as loaded_file:
                self.inputs[key] = loaded_file['data']
        
        target_filename = 'y_train.npz' if 'train' in self.prefix else 'y_test.npz'
        target_filepath = os.path.join(self.data_path, target_filename)
        with np.load(target_filepath, mmap_mode='r') as loaded_file:
            self.targets = loaded_file['data']

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X = {key: self.inputs[key][batch_indices] for key in self.input_keys}
        y = self.targets[batch_indices]
        if self.use_sample_weights:
            sample_weights = 1.0 + (y * self.weight_alpha)
            return X, y, sample_weights
        else:
            return X, y

# --- Custom Callbacks ---
class HistoryLogger(tf.keras.callbacks.Callback):
    """
    A custom callback to save the training history to a JSON file after each epoch.
    This prevents losing history if the training is interrupted.
    """
    def __init__(self, filepath):
        super(HistoryLogger, self).__init__()
        self.filepath = filepath
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        history_for_json = {key: [float(val) for val in values] for key, values in self.history.items()}
        with open(self.filepath, 'w') as f:
            json.dump(history_for_json, f)
        print(f" - History updated and saved to {self.filepath}")

# --- New: Live loss plotting ---
class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.history = {'loss': [], 'val_loss': []}
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))

        plt.figure(figsize=(8, 5))
        plt.plot(self.history['loss'], label='Train Loss', marker='o')
        plt.plot(self.history['val_loss'], label='Val Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'loss_curve_epoch_{epoch+1}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

# --- New: Validation sample predictions ---
class ValidationSampleCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, save_dir):
        super().__init__()
        self.val_gen = val_gen
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_gen[0][:2]
        y_pred = self.model.predict(X_val, verbose=0).ravel()
        y_val_sq = np.square(y_val)
        y_pred_sq = np.square(y_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_val_sq, y_pred_sq, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Affinity')
        plt.ylabel('Predicted Affinity')
        plt.title(f'Pred vs Actual (Epoch {epoch+1})')
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'val_scatter_epoch_{epoch+1}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

# --- Custom Model Layers ---
class PositionalEncoding(Layer):
    # Keras needs to know the arguments used to create the layer to save its config
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        # The positional encoding is created once and stored as a non-trainable weight
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_config(self):
        # This method tells Keras how to save the configuration of this layer
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embed_dim": self.embed_dim,
        })
        return config

    def positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

# --- Custom Loss Function ---
def create_weighted_mse(pos_weight=5.0, threshold=0.1):
    """
    Creates a custom Mean Squared Error loss function that applies a higher
    penalty to errors on high-affinity samples (y_true > threshold).
    This forces the model to focus on getting the important predictions right.
    """
    def weighted_mse(y_true, y_pred):
        mse_loss = tf.keras.losses.MeanSquaredError()
        mse = mse_loss(y_true, y_pred)
        weights = tf.where(y_true >= threshold, pos_weight, 1.0)
        return mse * weights
    return weighted_mse

# --- "Supreme" Model Architecture ---
def build_supreme_model(input_shapes, params):
    """
    Builds the hybrid CNN-LSTM-Attention-GNN model.
    It creates a parallel processing branch for each type of input data.
    """
    # Create an Input layer for each available data source
    input_layers = {key: Input(shape=shape, name=key) for key, shape in input_shapes.items()}
    
    # -- Reusable Processing Blocks --
    def create_seq_processor(input_tensor, p):
        # Adds positional info, then processes with CNN (for motifs) and Bi-LSTM (for context)
        max_len, features = input_tensor.shape[1], input_tensor.shape[2]
        pos_encoded_input = PositionalEncoding(max_len, features)(input_tensor)
        x = Conv1D(filters=p['cnn_filters'], kernel_size=p['cnn_kernel_size'], padding='same', activation='relu')(pos_encoded_input)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(p['lstm_units'], return_sequences=True))(x)
        return x

    def create_graph_processor(seq_input_tensor, adj_input_tensor, p):
        # Processes graph data (nodes + connections) using Graph Convolutional layers
        x = GCSConv(p['gnn_units'], activation='relu')([seq_input_tensor, adj_input_tensor])
        x = GCSConv(p['gnn_units'], activation='relu')([x, adj_input_tensor])
        return GlobalAveragePooling1D()(x)

    # -- Create all the processing branches --
    arch_params = params['model_architecture']
    primary_seq_processed = create_seq_processor(input_layers['primary_sequence_input'], arch_params)
    target_seq_processed = create_seq_processor(input_layers['target_sequence_input'], arch_params)
    competitor_seq_processed = create_seq_processor(input_layers['competitor_sequence_input'], arch_params)
    
    # The Attention mechanism learns relationships between the primary and target sequences
    attention_output = MultiHeadAttention(num_heads=arch_params['attention_heads'], key_dim=arch_params['lstm_units'])(query=primary_seq_processed, value=target_seq_processed, key=target_seq_processed)
    attention_output = LayerNormalization()(attention_output + primary_seq_processed)
    
    # Pool the features from all sequence branches into fixed-size vectors
    features_to_combine = [
        GlobalAveragePooling1D()(attention_output),
        GlobalAveragePooling1D()(target_seq_processed),
        GlobalAveragePooling1D()(competitor_seq_processed),
        GlobalAveragePooling1D()(Conv1D(32, 5, activation='relu')(input_layers['primary_structure_input'])),
        Dense(16, activation='relu')(input_layers['numerical_features_input'])
    ]

    # Conditionally add GNN branches only if graph data was provided
    gnn_params = params.get('gnn_architecture', {})
    if 'target_adjacency_input' in input_layers:
        print("  - Building GNN branch for Target molecule.")
        target_graph_features = create_graph_processor(input_layers['target_sequence_input'], input_layers['target_adjacency_input'], gnn_params)
        features_to_combine.append(target_graph_features)
    
    # -- Final Combination and Prediction --
    combined = concatenate(features_to_combine)
    combined = Dropout(params['dropout_rate'])(combined)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(combined)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    output = Dense(1, activation='sigmoid', name='affinity_output')(x)

    model = Model(inputs=input_layers, outputs=output)
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    config = load_config()
    params = config['training_parameters']
    
# --- Step 1: Setup Paths ---
    project_root = config['project_root']
    experiment_id = config.get('experiment_id', f"run_{int(time.time())}") # Use ID or a timestamp fallback

    # Create a dedicated folder for this experiment's outputs
    experiment_dir = os.path.join(project_root, 'experiments', experiment_id)
    model_save_dir = os.path.join(experiment_dir, config['output_folders']['main_models_folder'])
    logs_dir = os.path.join(experiment_dir, config['output_folders']['logs_subfolder'])
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])

    # --- Step 2: Prepare Data Generators ---
    print("\nPreparing data generators...")
    train_indices = np.arange(len(np.load(os.path.join(data_path, 'y_train.npz'))['data']))
    test_indices = np.arange(len(np.load(os.path.join(data_path, 'y_test.npz'))['data']))
    np.random.shuffle(train_indices)

    adv_params = params['advanced_training']
    model_inputs = params.get('model_inputs') # Get the list from config
    if not model_inputs:
        print("FATAL: 'model_inputs' key not found in training_parameters of config.json.")
        exit()

    train_generator = DataGenerator(data_path, params['batch_size'], train_indices, 'X_train_', adv_params, model_inputs)
    test_generator = DataGenerator(data_path, params['batch_size'], test_indices, 'X_test_', adv_params, model_inputs)

    # --- Step 3: Build and Compile Model ---
    print("\nBuilding the 'Supreme' regression model...")
    sample_X, _, _ = train_generator[0]
    input_shapes = {key: val.shape[1:] for key, val in sample_X.items()}
    
    model = build_supreme_model(input_shapes, params)
    
    # Select loss function based on config settings
    if adv_params['use_custom_loss']:
        loss_function = create_weighted_mse(adv_params['custom_loss_pos_weight'])
        print("  - Using custom weighted MSE loss function.")
    else:
        loss_function = 'mean_squared_error'

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss=loss_function, metrics=['mean_absolute_error'])
    model.summary()

    # --- Step 4: Define Callbacks ---
    print("\nDefining callbacks...")
    model_filepath = os.path.join(model_save_dir, 'best_supreme_model.keras')
    history_filepath = os.path.join(model_save_dir, 'history_supreme_model.json')
    log_dir = os.path.join(logs_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=adv_params.get('early_stopping_patience', 10), mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        TensorBoard(log_dir=log_dir),
        HistoryLogger(filepath=history_filepath), # Our custom history saver
         LivePlotCallback(save_dir=logs_dir),  # <-- inserted here
        ValidationSampleCallback(val_gen=test_generator, save_dir=logs_dir)  # <-- inserted here
    ]
    print(f"  - Model checkpoints will be saved to: {model_filepath}")
    print(f"  - TensorBoard logs will be saved to: {log_dir}")

    # --- Step 5: Start Training ---
    print("\nStarting model training...")
    training_start_time = time.time() # Timer Start
    model.fit(
        train_generator,
        epochs=params['epochs'],
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )
    training_end_time = time.time() # Timer End

    print("\n--- Supreme Model Training Complete ---")
    print(f"Total training time: {training_end_time - training_start_time:.2f} seconds")