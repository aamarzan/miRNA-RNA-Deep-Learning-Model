# quantize_model.py
import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
MODEL_DIR = 'model_files'
DATA_DIR = r'E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl' # Path to your .npy files
MODEL_NAME = 'supreme_model.keras'

# --- 1. Load your original Keras model ---
# You need to include your custom objects here, just like in app.py
# (This is a simplified version; add your PositionalEncoding class and weighted_mse function here if needed for loading)
print(f"Loading original model: {MODEL_NAME}...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME), compile=False)
print("Original model loaded successfully.")

# --- 2. Create a representative dataset for calibration ---
# Quantization requires a small sample of your training data to learn the range of values.
print("Loading representative data for quantization...")
def representative_dataset():
    # Load a small portion of your training data (e.g., 100 samples)
    # NOTE: The keys must match your model's input names
    x_train_cnn = np.load(os.path.join(DATA_DIR, 'X_train_cnn_lstm_input.npy'), mmap_mode='r')
    x_train_gnn = np.load(os.path.join(DATA_DIR, 'X_train_gnn_input.npy'), mmap_mode='r')
    x_train_num = np.load(os.path.join(DATA_DIR, 'X_train_numerical_input.npy'), mmap_mode='r')
    
    for i in range(100):
        yield {
            "cnn_lstm_input": x_train_cnn[i:i+1],
            "gnn_input": x_train_gnn[i:i+1],
            "numerical_input": x_train_num[i:i+1],
        }

# --- 3. Convert the model to TensorFlow Lite and Quantize it ---
print("Converting and quantizing model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# Ensure that the converter uses integer operations for inputs and outputs
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()
print("Quantization complete.")

# --- 4. Save the new, smaller model ---
quantized_model_path = os.path.join(MODEL_DIR, 'supreme_model_quantized.tflite')
with open(quantized_model_path, 'wb') as f:
    f.write(tflite_model_quant)

print(f"Successfully saved quantized model to: {quantized_model_path}")
print(f"Original size: {os.path.getsize(os.path.join(MODEL_DIR, MODEL_NAME))/1e6:.2f} MB")
print(f"New size: {os.path.getsize(quantized_model_path)/1e6:.2f} MB")