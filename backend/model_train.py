import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import both cleaning functions
from preprocess import load_data, simple_clean, enhanced_clean

# ===================================================================
# CONFIGURATION
# ===================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_brands.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

TOP_WORDS = 15000 # Number of most frequent words to keep
MAXLEN = 120    # Max sequence length for padding
EMBEDDING_DIM = 128 # Dimension of word embeddings
LSTM_UNITS = 128    # Units in the LSTM layer
BATCH_SIZE = 256    # Batch size for training
EPOCHS = 10         # Maximum number of epochs
USE_ENHANCED_CLEANING = False # Set to True if you want to use enhanced_clean
CLEANING_FUNCTION = enhanced_clean if USE_ENHANCED_CLEANING else simple_clean
print(f"Using cleaning function: {CLEANING_FUNCTION.__name__}")

# ===================================================================
# 1. LOAD AND PREPROCESS DATA
# ===================================================================
print("\nLoading and preprocessing data...")
df = load_data(DATA_PATH)
# For sentiment model, we'll typically exclude neutral for binary classification
# Or, if you want 3 classes, adjust `label` creation and `num_classes`
df_train = df[df['sentiment'] != 'neutral'].copy() # Focus on clear positive/negative for binary
df_train['label'] = (df_train['sentiment'] == 'positive').astype(int) # 1 for positive, 0 for negative
df_train['clean_review'] = df_train['Reviews'].apply(CLEANING_FUNCTION)
print("Data loaded and cleaned.")

# ===================================================================
# 2. TRAIN-VALIDATION SPLIT & TOKENIZATION
# ===================================================================
print("\nSplitting and tokenizing data...")
X_train, X_val, y_train, y_val = train_test_split(
    df_train['clean_review'], df_train['label'], test_size=0.15, random_state=42, stratify=df_train['label']
)
tokenizer = Tokenizer(num_words=TOP_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAXLEN, padding='post')
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=MAXLEN, padding='post')
y_train_cat = to_categorical(y_train, num_classes=2) # 2 classes: positive and negative
y_val_cat = to_categorical(y_val, num_classes=2)
print("Tokenization complete.")

# ===================================================================
# 3. BUILD THE GPU-OPTIMIZED LSTM MODEL
# ===================================================================
print("\nBuilding the GPU-optimized model architecture...")
model = Sequential([
    Embedding(input_dim=TOP_WORDS, output_dim=EMBEDDING_DIM, input_length=MAXLEN),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=False, dropout=0.2)), # Bi-LSTM layer
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax') # Output for 2 classes
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ===================================================================
# 4. TRAIN THE MODEL
# ===================================================================
print("\nStarting model training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'lstm_model.keras'), save_best_only=True, monitor='val_loss')
]
history = model.fit(
    X_train_seq, y_train_cat,
    validation_data=(X_val_seq, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
print("Model training finished.")

# ===================================================================
# 5. SAVE TOKENIZER AND PLOT RESULTS
# ===================================================================
print("\nSaving tokenizer and plotting training history...")
with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plot_path = os.path.join(MODEL_DIR, 'training_history.png')
plt.savefig(plot_path)
print(f"Training history plot saved to {plot_path}")

print("\n--- Training script finished successfully! ---")