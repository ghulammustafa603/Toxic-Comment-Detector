"""
Script to create and save the tokenizer from training data
This is useful when the model exists but tokenizer.pkl is missing
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

print("=" * 60)
print("Creating Tokenizer from Training Data")
print("=" * 60)

# Load training data
train_path = "data/train.csv"

if not os.path.exists(train_path):
    print(f"❌ Error: Dataset not found at {train_path}")
    print("Please ensure the train.csv file is in the data/ directory.")
    exit(1)

print("\n1. Loading training data...")
train = pd.read_csv(train_path)
print(f"   ✓ Loaded {len(train)} comments")

# Clean text
print("\n2. Cleaning text...")
train["clean_text"] = train["comment_text"].apply(clean_text)
print("   ✓ Text cleaned")

# Prepare features
X = train["clean_text"].values

# Tokenization parameters (must match training parameters)
max_words = 20000
max_len = 200

print("\n3. Creating and fitting tokenizer...")
print(f"   - Max words: {max_words}")
print(f"   - Max length: {max_len}")

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)

print(f"   ✓ Tokenizer created with {len(tokenizer.word_index)} unique words")

# Save tokenizer
print("\n4. Saving tokenizer...")
with open("saved_model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("   ✓ Tokenizer saved to saved_model/tokenizer.pkl")

# Save tokenizer info
tokenizer_info = {
    "num_words": max_words,
    "max_len": max_len,
    "word_index_size": len(tokenizer.word_index)
}
with open("saved_model/tokenizer_info.pkl", "wb") as f:
    pickle.dump(tokenizer_info, f)
print("   ✓ Tokenizer info saved to saved_model/tokenizer_info.pkl")

print("\n" + "=" * 60)
print("✓ Tokenizer created successfully!")
print("=" * 60)
print("\nYou can now run the Streamlit app:")
print("  streamlit run app.py")
print("=" * 60)

