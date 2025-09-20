import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from spektral.layers import GCSConv
import shutil

# --- 1. DYNAMIC PATH CONFIGURATION ---
print("--- Locating project folders...")
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    MODEL_DIR = os.path.join(project_root, 'models')
    KERAS_MODEL_NAME = 'supreme_model.keras'
    TFLITE_MODEL_NAME = 'supreme_model_quantized.tflite'
    TEMP_SAVEDMODEL_DIR = os.path.join(project_root, 'temp_saved_model')
    print(f"  - Project Root identified as: {project_root}")
    print(f"  - Model Directory set to: {MODEL_DIR}")
except Exception as e:
    print(f"FATAL ERROR: Could not determine file paths. Error: {e}")
    exit()

print("\n--- Starting Model Quantization Process ---")

# --- 2. DEFINE CUSTOM OBJECTS ---
class PositionalEncoding(Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
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
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

def weighted_mse(y_true, y_pred):
    pos_weight = 10.0
    threshold = 0.1
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse = mse_loss(y_true, y_pred)
    weights = tf.where(y_true >= threshold, pos_weight, 1.0)
    return mse * weights

# --- 3. LOAD THE ORIGINAL KERAS MODEL ---
print("\nStep 1: Loading the original .keras model...")
try:
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'weighted_mse': weighted_mse,
        'GCSConv': GCSConv
    }
    keras_model_path = os.path.join(MODEL_DIR, KERAS_MODEL_NAME)
    model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects)
    print("Original model loaded successfully.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load the Keras model. Error: {e}")
    exit()

# --- 4. CONVERT AND QUANTIZE THE MODEL (Final Combined Method) ---
print("\nStep 2: Converting model using SavedModel, Float16, and Select TF Ops...")
try:
    # Step 4a: Export the model to the temporary SavedModel format.
    if os.path.exists(TEMP_SAVEDMODEL_DIR):
        shutil.rmtree(TEMP_SAVEDMODEL_DIR)
    model.export(TEMP_SAVEDMODEL_DIR)
    print("Model successfully exported to intermediate SavedModel format.")

    # Step 4b: Initialize the converter from the SavedModel directory.
    converter = tf.lite.TFLiteConverter.from_saved_model(TEMP_SAVEDMODEL_DIR)

    # Step 4c: Apply optimizations.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # --- THIS IS THE FINAL COMBINED FIX ---
    # Use Float16 quantization.
    converter.target_spec.supported_types = [tf.float16]
    # AND allow Select TF Ops for maximum compatibility with LSTM layers.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    # --- END OF FIX ---

    print("Starting conversion process...")
    tflite_model_quant = converter.convert()
    print("Model Float16 quantization complete.")

except Exception as e:
    print(f"\nFATAL ERROR: The TFLite conversion failed.")
    print(f"Error details: {e}")
    exit()
finally:
    # Clean up the temporary directory
    if os.path.exists(TEMP_SAVEDMODEL_DIR):
        shutil.rmtree(TEMP_SAVEDMODEL_DIR)

# --- 5. SAVE THE NEW, SMALLER MODEL ---
print("\nStep 3: Saving the new .tflite model...")
tflite_model_path = os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model_quant)

print("\n--- PROCESS COMPLETE ---")
print(f"Successfully saved quantized model to: {tflite_model_path}")

original_size_mb = os.path.getsize(keras_model_path) / (1024 * 1024)
new_size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)

print(f"\nOriginal model size: {original_size_mb:.2f} MB")
print(f"New quantized model size: {new_size_mb:.2f} MB")
print(f"Size reduction: {((original_size_mb - new_size_mb) / original_size_mb) * 100:.1f}%")