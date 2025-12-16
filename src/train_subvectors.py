import keras
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, Concatenate
from tensorflow.keras.optimizers import Nadam
#from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU(s) available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            print(f"  Memory Growth: Enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU Error: {e}")
        print("Falling back to CPU...")
else:
    print("❌ No GPU available, using CPU for training")

print(f"🔧 TensorFlow Version: {tf.__version__}")
print(f"🖥️ Device: {'GPU' if len(gpus) > 0 else 'CPU'}")
print("-" * 50)

# --- Subvector Configuration ---
SUBVECTOR_CONFIG = {
    'location_l_hand': 63,
    'location_r_hand': 63,
    'location_pose': 48,      # UPDATED: Upper body only (no legs)
    'handshape_l': 210,
    'handshape_r': 210,
    'palm_orientation': 200,
    'movement': 126
}

TOTAL_FEATURES = sum(SUBVECTOR_CONFIG.values())  # Now 920 instead of 947
print(f"\n📊 Subvector Configuration:")
for name, size in SUBVECTOR_CONFIG.items():
    print(f"  {name}: {size} features")
print(f"  Total: {TOTAL_FEATURES} features")
print("-" * 50)

# --- 1. Load and Prepare Data ---
NP_DATA_PATH = os.path.join('data') 
MODEL_DIR = 'model'
REPORT_DIR = 'report'
LOGS_DIR = 'logs'

# Ensure required directories exist
for directory in [MODEL_DIR, REPORT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

if not os.path.exists(NP_DATA_PATH):
    raise FileNotFoundError(f"Data directory not found at '{NP_DATA_PATH}'. Please run the preprocessing script first.")

actions = [d for d in os.listdir(NP_DATA_PATH) if os.path.isdir(os.path.join(NP_DATA_PATH, d))]
label_map = {label: num for num, label in enumerate(actions)}

print(f"\nFound gestures: {actions}")

sequences, labels = [], []
for action in actions:
    action_path = os.path.join(NP_DATA_PATH, action)
    sequence_files = os.listdir(action_path)
    for sequence_file in sequence_files:
        res = np.load(os.path.join(action_path, sequence_file))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Verify data dimensions
print(f"\nData shape: {X.shape}")
if X.shape[-1] != TOTAL_FEATURES:
    print(f"⚠️ WARNING: Expected {TOTAL_FEATURES} features but got {X.shape[-1]}")
    print("Please verify your subvector configuration matches your data!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# --- 2. Split Data into Subvectors ---
def split_into_subvectors(X):
    """Split input tensor into 7 semantic subvectors"""
    subvectors = []
    start_idx = 0
    
    for name, size in SUBVECTOR_CONFIG.items():
        end_idx = start_idx + size
        subvector = X[:, :, start_idx:end_idx]  # (batch, sequence, features)
        subvectors.append(subvector)
        start_idx = end_idx
    
    return subvectors

# Split training and test data
X_train_subvectors = split_into_subvectors(X_train)
X_test_subvectors = split_into_subvectors(X_test)

print(f"\n✅ Data split into {len(X_train_subvectors)} subvectors:")
for i, (name, size) in enumerate(SUBVECTOR_CONFIG.items()):
    print(f"  Subvector {i+1} ({name}): {X_train_subvectors[i].shape}")

# --- 3. Build Multi-Stream Transformer Model ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Creates a single Transformer Encoder block."""
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def build_multi_stream_transformer(subvector_shapes, head_size, num_heads, ff_dim, 
                                   num_transformer_blocks, num_classes, dropout=0):
    """
    Builds a multi-stream Transformer model that processes each subvector independently
    before combining them for classification.
    """
    inputs = []
    encoded_streams = []
    
    # Create separate processing stream for each subvector
    for i, (name, shape) in enumerate(zip(SUBVECTOR_CONFIG.keys(), subvector_shapes)):
        # Input for this subvector
        inp = Input(shape=shape, name=f'input_{name}')
        inputs.append(inp)
        
        # Apply transformer encoder blocks to this stream
        x = inp
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        
        # Global average pooling for this stream
        x = GlobalAveragePooling1D(data_format="channels_last", name=f'pool_{name}')(x)
        encoded_streams.append(x)
    
    # Concatenate all encoded streams
    if len(encoded_streams) > 1:
        merged = Concatenate(name='merge_streams')(encoded_streams)
    else:
        merged = encoded_streams[0]
    
    # Classification head
    x = Dropout(0.4)(merged)
    x = Dense(256, activation="relu", name='fc1')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu", name='fc2')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax", name='output')(x)

    return Model(inputs=inputs, outputs=outputs, name='MultiStreamTransformer')

# --- Hyperparameters ---
SEQUENCE_LENGTH = X_train.shape[1]
HEAD_SIZE = 64  # Reduced since we have multiple streams
NUM_HEADS = 4   # previously 2
FF_DIM = 128    # previously 4
NUM_TRANSFORMER_BLOCKS = 2
NUM_CLASSES = len(actions)
DROPOUT = 0.2
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

# Get shapes for each subvector
subvector_shapes = [(SEQUENCE_LENGTH, size) for size in SUBVECTOR_CONFIG.values()]

# Build the multi-stream model
model = build_multi_stream_transformer(
    subvector_shapes,
    head_size=HEAD_SIZE,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

model.compile(
    optimizer=Nadam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#model.compile(
#    optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01),
#    loss='categorical_crossentropy',
#    metrics=['accuracy']
#)

print("\n" + "="*50)
print("Model Architecture Summary")
print("="*50)
model.summary()

# --- 4. Setup Enhanced Callbacks ---
log_base_dir = 'Logs_MultiStream_TF'
# Clean up old log directories (keep last 3)
try:
    if os.path.exists(log_base_dir):
        existing_logs = sorted([d for d in os.listdir(log_base_dir) 
                           if os.path.isdir(os.path.join(log_base_dir, d))])
        if len(existing_logs) > 3:
            for old_log in existing_logs[:-3]:  # Keep last 3
                import shutil
                old_log_path = os.path.join(log_base_dir, old_log)
                shutil.rmtree(old_log_path, ignore_errors=True)
                print(f"🧹 Cleaned up old log: {old_log}")
except Exception as e:
    print(f"⚠️ Warning: Could not clean up old logs: {e}")

# Use readable timestamp for log directory
timestamp = time.strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join(log_base_dir, f'training_run_{timestamp}')
os.makedirs(log_dir, exist_ok=True)

tb_callback = TensorBoard(log_dir=log_dir)

early_stopping_callback = EarlyStopping(
    monitor='val_loss', 
    patience=15,  # Reduced from 20 since AdamW with 0.0001 should converge faster
    restore_best_weights=True,
    verbose=1
)

checkpoint_callback = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'bigger.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# --- 5. Train the Model ---
print("\n" + "="*50)
print("Starting Training with Multi-Stream Architecture...")
print("="*50 + "\n")

start_time = time.time()

history = model.fit(
    X_train_subvectors, y_train,  # Pass list of subvectors
    epochs=500,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_subvectors, y_test),
    callbacks=[tb_callback, early_stopping_callback, checkpoint_callback, reduce_lr_callback]
)

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# --- 6. Evaluate and Generate Reports ---
print("\n" + "="*50)
print("Evaluating Model...")
print("="*50 + "\n")

# Load best model for evaluation
from tensorflow.keras.models import load_model
try:
    best_model = load_model(os.path.join(MODEL_DIR, 'bigger.h5'))
    print("✅ Best model loaded successfully")
except Exception as e:
    print(f"❌ Error loading best model: {e}")
    print("Using current model for evaluation...")
    best_model = model

# Make predictions
y_pred = best_model.predict(X_test_subvectors)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\n📊 Classification Report:")
print("="*50)
print(classification_report(y_true_classes, y_pred_classes, target_names=actions))

# Generate and save confusion matrix
try:
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix - Multi-Stream Transformer', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    confusion_path = os.path.join(REPORT_DIR, 'bigger_confusion.png')
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved as '{confusion_path}'")
    plt.close()
except Exception as e:
    print(f"\n❌ Error saving confusion matrix: {e}")
    plt.close()

# --- 7. Plot Training History ---
def plot_training_history(history):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        history_path = os.path.join(REPORT_DIR, 'bigger.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved as '{history_path}'")
    except Exception as e:
        print(f"❌ Error saving training history plot: {e}")
    finally:
        plt.close()

plot_training_history(history)

# --- 8. Save Training Configuration ---
training_config = {
    'architecture': 'Multi-Stream Transformer',
    'subvectors': SUBVECTOR_CONFIG,
    'model_architecture': {
        'head_size': HEAD_SIZE,
        'num_heads': NUM_HEADS,
        'ff_dim': FF_DIM,
        'num_transformer_blocks': NUM_TRANSFORMER_BLOCKS,
        'dropout': DROPOUT,
    },
    'training_parameters': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs_run': len(history.history['loss']),
        'optimizer': 'Nadam',
    },
    'results': {
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(min(history.history['val_loss']))
    },
    'metadata': {
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'actions': actions,
        'num_classes': NUM_CLASSES,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'data_shape': list(X.shape),
        'total_features': TOTAL_FEATURES,
        'tensorflow_version': tf.__version__,
        'device': 'GPU' if len(gpus) > 0 else 'CPU'
    }
}

try:
    config_path = os.path.join(REPORT_DIR, 'bigger.json')
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)
    print(f"\n✅ Training configuration saved as '{config_path}'")
except Exception as e:
    print(f"\n❌ Error saving training configuration: {e}")

# --- 9. Save Final Model ---
try:
    final_model_path = os.path.join(MODEL_DIR, 'bigger_final.h5')
    model.save(final_model_path)
    print(f"\n✅ Final model saved as '{final_model_path}'")
    print(f"✅ Best model saved as '{os.path.join(MODEL_DIR, 'bigger.h5')}'")
except Exception as e:
    print(f"\n❌ Error saving final model: {e}")

# --- Summary ---
print("\n" + "="*50)
print("Training Summary - Multi-Stream Transformer")
print("="*50)
print(f"Architecture: 7 parallel streams (one per subvector)")
print(f"Subvectors: {list(SUBVECTOR_CONFIG.keys())}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Total Training Time: {training_time/60:.2f} minutes")
print(f"Epochs Completed: {len(history.history['loss'])}")
print("\n📁 Generated Files:")
print("  - bigger.h5 (best performing model)")
print("  - bigger_final.h5 (final model)")
print("  - bigger_confusion.png")
print("  - bigger.png")
print("  - bigger.json")
print("\n💡 To view TensorBoard logs, run:")
print(f"   tensorboard --logdir={log_dir}")
print("="*50)
