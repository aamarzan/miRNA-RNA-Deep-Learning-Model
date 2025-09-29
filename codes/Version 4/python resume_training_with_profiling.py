# resume_training_with_profiling.py
import os
import json
import numpy as np
import tensorflow as tf
from s3b_build_model import DataGenerator, build_supreme_model, create_weighted_mse, PositionalEncoding

# --- Load config ---
with open("E:/1. Github/1. miRNA-RNA-Deep-Learning-Model/codes/Version 4/config.json", "r") as f:

    config = json.load(f)

params = config['training_parameters']
adv_params = params['advanced_training']
project_root = config['project_root']
experiment_id = config['experiment_id']

# Paths
experiment_dir = os.path.join(project_root, 'experiments', experiment_id)
model_dir = os.path.join(experiment_dir, config['output_folders']['main_models_folder'])
data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])

# --- Load model ---
custom_objects = {'PositionalEncoding': PositionalEncoding}
if adv_params['use_custom_loss']:
    custom_objects['weighted_mse'] = create_weighted_mse(adv_params['custom_loss_pos_weight'])

model_path = os.path.join(model_dir, 'best_supreme_model.keras')
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# --- Prepare data generators ---
train_indices = np.arange(len(np.load(os.path.join(data_path, 'y_train.npz'))['data']))
test_indices = np.arange(len(np.load(os.path.join(data_path, 'y_test.npz'))['data']))
np.random.shuffle(train_indices)

model_inputs = params['model_inputs']
train_gen = DataGenerator(data_path, params['batch_size'], train_indices, 'X_train_', adv_params, model_inputs)
test_gen = DataGenerator(data_path, params['batch_size'], test_indices, 'X_test_', adv_params, model_inputs)

# --- Profiling setup ---
tf.config.experimental_run_functions_eagerly(True)
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

# --- Callbacks ---
resume_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_supreme_model_resumed.keras'),
        save_best_only=True, monitor='val_loss', mode='min', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=adv_params.get('early_stopping_patience', 10),
        mode='min', restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(experiment_dir, config['output_folders']['logs_subfolder'], "resume_fit")
    )
]

# --- Resume training ---
print("\nResuming training from Epoch 71...")
model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=params['epochs'],
    initial_epoch=71,
    callbacks=resume_callbacks,
    verbose=1
)

print("\n--- Resume Training Complete ---")
