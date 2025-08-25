# 3_model_building.py (Definitive, Final Version)
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
import datetime
from spektral.layers import GCSConv

# --- Configuration Loader ---
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

# --- Custom Data Generator (Final, Robust Version) ---
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size, indices, prefix, advanced_params):
        self.data_path = data_path
        self.batch_size = batch_size
        self.indices = indices
        self.prefix = prefix
        self.use_sample_weights = advanced_params.get('use_sample_weighting', False)
        self.weight_alpha = advanced_params.get('sample_weight_alpha', 10.0)
        
        # <<< CHANGE: Using the robust, explicit file-checking logic from our sanity check >>>
        # Define all *potential* input keys we might have now or in the future
        potential_keys = [
            'primary_sequence_input',
            'primary_structure_input',
            'target_sequence_input',
            'competitor_sequence_input',
            'numerical_features_input',
            'primary_adjacency_input',
            'target_adjacency_input',
            'competitor_adjacency_input'
        ]
        
        self.input_keys = []
        print(f"  - DataGenerator searching for inputs in: {data_path}")
        for key in potential_keys:
            filepath = os.path.join(self.data_path, f'{self.prefix}{key}.npy')
            if os.path.exists(filepath):
                self.input_keys.append(key)
        
        if not self.input_keys:
            raise FileNotFoundError(f"No input .npy files starting with '{self.prefix}' found in '{data_path}'.")
        print(f"  - Found {len(self.input_keys)} input sources: {sorted(self.input_keys)}")

        self.inputs = {key: np.load(os.path.join(data_path, f'{self.prefix}{key}.npy'), mmap_mode='r') for key in self.input_keys}
        
        target_filename = 'y_train.npy' if 'train' in self.prefix else 'y_test.npy'
        self.targets = np.load(os.path.join(self.data_path, target_filename), mmap_mode='r')

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

# <<< NEW: Custom Layer for Positional Encoding >>>
class PositionalEncoding(Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_config(self):
        config = super().get_config()
        return config

    def positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        # x shape is (batch, seq_len, features)
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

# --- Custom Weighted Loss Function (Corrected for new TensorFlow/Keras version) ---
def create_weighted_mse(pos_weight=5.0, threshold=0.1):
    def weighted_mse(y_true, y_pred):
        # <<< FIX: Use the class-based MeanSquaredError as recommended by the error message >>>
        mse_loss = tf.keras.losses.MeanSquaredError()
        mse = mse_loss(y_true, y_pred)
        
        # Apply higher penalty for errors on high-affinity samples
        weights = tf.where(y_true >= threshold, pos_weight, 1.0)
        return mse * weights
    return weighted_mse

def build_supreme_model(input_shapes, params):
    input_layers = {key: Input(shape=shape, name=key) for key, shape in input_shapes.items()}
    
    # <<< CHANGE: Positional Encoding is now applied to sequence inputs >>>
    def create_seq_processor(input_tensor, p):
        # The PositionalEncoding layer gives the model information about the position of each nucleotide
        max_len, features = input_tensor.shape[1], input_tensor.shape[2]
        pos_encoded_input = PositionalEncoding(max_len, features)(input_tensor)
        
        x = Conv1D(filters=p['cnn_filters'], kernel_size=p['cnn_kernel_size'], padding='same', activation='relu')(pos_encoded_input)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(p['lstm_units'], return_sequences=True))(x)
        return x

    def create_graph_processor(seq_input_tensor, adj_input_tensor, p):
        x = GCSConv(p['gnn_units'], activation='relu')([seq_input_tensor, adj_input_tensor])
        x = GCSConv(p['gnn_units'], activation='relu')([x, adj_input_tensor])
        return GlobalAveragePooling1D()(x)

    arch_params = params['model_architecture']
    primary_seq_processed = create_seq_processor(input_layers['primary_sequence_input'], arch_params)
    target_seq_processed = create_seq_processor(input_layers['target_sequence_input'], arch_params)
    competitor_seq_processed = create_seq_processor(input_layers['competitor_sequence_input'], arch_params)
    
    attention_output = MultiHeadAttention(num_heads=arch_params['attention_heads'], key_dim=arch_params['lstm_units'])(query=primary_seq_processed, value=target_seq_processed, key=target_seq_processed)
    attention_output = LayerNormalization()(attention_output + primary_seq_processed)
    
    features_to_combine = [
        GlobalAveragePooling1D()(attention_output),
        GlobalAveragePooling1D()(target_seq_processed),
        GlobalAveragePooling1D()(competitor_seq_processed),
        GlobalAveragePooling1D()(Conv1D(32, 5, activation='relu')(input_layers['primary_structure_input'])),
        Dense(16, activation='relu')(input_layers['numerical_features_input'])
    ]

    gnn_params = params.get('gnn_architecture', {})
    if 'target_adjacency_input' in input_layers:
        print("  - Building GNN branch for Target molecule.")
        target_graph_features = create_graph_processor(input_layers['target_sequence_input'], input_layers['target_adjacency_input'], gnn_params)
        features_to_combine.append(target_graph_features)
    
    combined = concatenate(features_to_combine)
    combined = Dropout(params['dropout_rate'])(combined)
    
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='affinity_output')(x)

    model = Model(inputs=input_layers, outputs=output)
    return model

if __name__ == "__main__":
    config = load_config()
    params = config['training_parameters']
    
    project_root = config['project_root']
    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    model_save_dir = os.path.join(project_root, config['output_folders']['main_models_folder'])
    logs_dir = os.path.join(project_root, config['output_folders']['logs_subfolder'])
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\nStep 1: Getting data indices...")
    train_indices = np.arange(len(np.load(os.path.join(data_path, 'y_train.npy'))))
    test_indices = np.arange(len(np.load(os.path.join(data_path, 'y_test.npy'))))
    np.random.shuffle(train_indices)

    print("\nStep 2: Creating data generators...")
    adv_params = params['advanced_training']
    train_generator = DataGenerator(data_path, params['batch_size'], train_indices, 'X_train_', adv_params)
    test_generator = DataGenerator(data_path, params['batch_size'], test_indices, 'X_test_', adv_params)

    print("\nStep 3: Building the 'Supreme' regression model...")
    sample_X, _, _ = train_generator[0]
    input_shapes = {key: val.shape[1:] for key, val in sample_X.items()}
    
    model = build_supreme_model(input_shapes, params)
    
    if params['advanced_training']['use_custom_loss']:
        loss_function = create_weighted_mse(params['advanced_training']['custom_loss_pos_weight'])
        print("  - Using custom weighted MSE loss function.")
    else:
        loss_function = 'mean_squared_error'

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss=loss_function, metrics=['mean_absolute_error'])
    model.summary()

    print("\nStep 4: Defining callbacks...")
    model_filepath = os.path.join(model_save_dir, 'best_supreme_model.keras')
    history_filepath = os.path.join(model_save_dir, 'history_supreme_model.json')
    log_dir = os.path.join(logs_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=adv_params.get('early_stopping_patience', 10), mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        TensorBoard(log_dir=log_dir)
    ]

    print("\nStep 5: Starting model training...")
    history = model.fit(train_generator, epochs=params['epochs'], validation_data=test_generator, callbacks=callbacks, verbose=1)

    print(f"\nStep 6: Saving training history...")
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(history_filepath, 'w') as f:
        json.dump(history_dict, f)

    print("\n--- Supreme Model Training Complete ---")