"""
Training script for Toxic Comment Detector Model
This script trains the LSTM model and saves it with the tokenizer
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pathlib import Path

# Create saved_model directory if it doesn't exist
os.makedirs("saved_model", exist_ok=True)

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text)
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)  # only alphabets
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

def train_model():
    """Train the toxic comment detector model"""
    
    print("Loading data...")
    # Load training data
    train_path = "data/train.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: Dataset not found at {train_path}")
        print("Please ensure the train.csv file is in the data/ directory.")
        return False
    
    train = pd.read_csv(train_path)
    print(f"Dataset loaded: {len(train)} comments")
    
    # Clean text
    print("Cleaning text...")
    train["clean_text"] = train["comment_text"].apply(clean_text)
    
    # Prepare features and labels
    X = train["clean_text"].values
    y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Tokenization
    print("Tokenizing text...")
    max_words = 20000
    max_len = 200
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Padding
    print("Padding sequences...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Build model
    print("Building model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, 128, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation='sigmoid')  # 6 labels for multi-label classification
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_pad, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    print("\nSaving model and tokenizer...")
    model.save("saved_model/toxic_lstm.h5")
    print("✓ Model saved to saved_model/toxic_lstm.h5")
    
    # Save tokenizer
    with open("saved_model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("✓ Tokenizer saved to saved_model/tokenizer.pkl")
    
    # Save tokenizer info
    tokenizer_info = {
        "num_words": max_words,
        "max_len": max_len,
        "word_index_size": len(tokenizer.word_index)
    }
    with open("saved_model/tokenizer_info.pkl", "wb") as f:
        pickle.dump(tokenizer_info, f)
    print("✓ Tokenizer info saved to saved_model/tokenizer_info.pkl")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Toxic Comment Detector - Model Training")
    print("=" * 60)
    
    success = train_model()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
        print("\nYou can now run the Streamlit app with:")
        print("  streamlit run app.py")
    else:
        print("\n" + "=" * 60)
        print("✗ Training failed!")
        print("=" * 60)
