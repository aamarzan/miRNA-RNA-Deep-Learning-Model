# s3b_build_model.py — Enhanced with LivePlotCallback & ValidationSampleCallback

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

# ---------------- CONFIG LOADER ----------------
def load_config(config_path=None):
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

# ---------------- DATA GENERATOR ----------------
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size, indices, prefix, advanced_params, model_inputs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.indices = indices
        self.prefix = prefix
        self.use_sample_weights = advanced_params.get('use_sample_weighting', False)
        self.weight_alpha = advanced_params.get('sample_weight_alpha', 10.0)

        potential_keys = model_inputs
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

# ---------------- CUSTOM CALLBACKS ----------------
class HistoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
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

# ---------------- CUSTOM LAYERS ----------------
class PositionalEncoding(Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "embed_dim": self.embed_dim})
        return config

    def positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

# ---------------- LOSS FUNCTION ----------------
def create_weighted_mse(pos_weight=5.0, threshold=0.1):
    def weighted_mse(y_true, y_pred):
        mse_loss = tf.keras.losses.MeanSquaredError()
        mse = mse_loss(y_true, y_pred)
        weights = tf.where(y_true >= threshold, pos_weight, 1.0)
        return mse * weights
    return weighted_mse

# ---------------- MODEL ARCHITECTURE ----------------
def build_supreme_model(input_shapes, params):
    input_layers = {key: Input(shape=shape, name=key) for key, shape in input_shapes.items()}

    def create_seq_processor(input_tensor, p):
        max_len, features = input_tensor.shape[1], input_tensor.shape[2]
        pos_encoded_input = PositionalEncoding(max_len, features)(input_tensor)
        x = Conv1D(filters=p['cnn_filters'], kernel_size=p['cnn_kernel_size'], padding='same', activation='relu')(pos_encoded_input)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(p['lstm_units'], return_sequences=True))(x)
        if p.get('use_attention', False):
            attn_out = MultiHeadAttention(num_heads=p.get('attention_heads', 4),
                                          key_dim=p.get('attention_key_dim', 32))(x, x)
            x = LayerNormalization()(x + attn_out)
        return x

    # Process each input branch
    seq_params = params.get('sequence_branch_params', {})
    processed_branches = []
    for key, input_layer in input_layers.items():
        if 'sequence' in key:
            processed_branches.append(create_seq_processor(input_layer, seq_params))
        elif 'structure' in key:
            # Structure branch: simple Conv + Pool
            s = Conv1D(filters=seq_params['cnn_filters'], kernel_size=3, padding='same', activation='relu')(input_layer)
            s = BatchNormalization()(s)
            processed_branches.append(s)
        elif 'numerical' in key:
            processed_branches.append(input_layer)

    # Merge branches
    merged_seq = concatenate(
        [GlobalAveragePooling1D()(b) if len(b.shape) == 3 else b for b in processed_branches]
    )

    # Dense layers after merge
    x = Dense(params.get('dense_units', 128), activation='relu', kernel_regularizer=l2(1e-4))(merged_seq)
    x = Dropout(params.get('dropout_rate', 0.3))(x)
    x = Dense(params.get('dense_units_2', 64), activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(params.get('dropout_rate', 0.3))(x)

    # Output layer
    output = Dense(1, activation='linear', name='affinity_output')(x)

    model = Model(inputs=list(input_layers.values()), outputs=output)
    return model


def main():
    start_time = time.time()
    config = load_config()
    params = {**config['processing_parameters'], **config['training_parameters']}
    project_root = config['project_root']
    processed_dl_folder = os.path.join(project_root, config['data_folders']['main_dataset_folder'],
                                       config['data_folders']['processed_for_dl_subfolder'])

    # Detect available input shapes
    model_inputs = params.get('model_inputs', [])
    input_shapes = {}
    for key in model_inputs:
        npz_path = os.path.join(processed_dl_folder, f'X_train_{key}.npz')
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                input_shapes[key] = data['data'].shape[1:]
    print(f"Detected input shapes: {input_shapes}")

    # Build model
    model = build_supreme_model(input_shapes, params)
    model.summary()

    # Compile model
    loss_fn = create_weighted_mse(pos_weight=params.get('pos_weight', 5.0),
                                  threshold=params.get('weight_threshold', 0.1)) \
              if params.get('use_sample_weighting', False) else 'mse'
    model.compile(optimizer=Adam(learning_rate=params.get('learning_rate', 1e-3)),
                  loss=loss_fn,
                  metrics=['mse'])

    # Prepare data generators
    train_indices = np.arange(len(np.load(os.path.join(processed_dl_folder, 'y_train.npz'))['data']))
    test_indices = np.arange(len(np.load(os.path.join(processed_dl_folder, 'y_test.npz'))['data']))

    train_gen = DataGenerator(processed_dl_folder, params['batch_size'], train_indices, 'X_train_', params, model_inputs)
    test_gen = DataGenerator(processed_dl_folder, params['batch_size'], test_indices, 'X_test_', params, model_inputs)

    # Logging and checkpoints
    logs_dir = os.path.join(project_root, 'training_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logs_dir, exist_ok=True)
    history_filepath = os.path.join(logs_dir, 'history_supreme_model.json')

    callbacks = [
        ModelCheckpoint(os.path.join(logs_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=params.get('patience', 10), restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        TensorBoard(log_dir=logs_dir),
        HistoryLogger(filepath=history_filepath),
        LivePlotCallback(save_dir=logs_dir),
        ValidationSampleCallback(val_gen=test_gen, save_dir=logs_dir)
    ]

    # Train
    history = model.fit(train_gen,
                        validation_data=test_gen,
                        epochs=params.get('epochs', 50),
                        callbacks=callbacks,
                        verbose=1)

    # Save final model
    final_model_path = os.path.join(logs_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    print(f"Total training time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
