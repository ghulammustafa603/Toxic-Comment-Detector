"""
Quick script to save model and tokenizer from notebook
Run this AFTER you've trained the model in your notebook
"""

import os
import sys
import pickle

# Make sure saved_model directory exists
os.makedirs("saved_model", exist_ok=True)

print("=" * 60)
print("Quick Model & Tokenizer Saver")
print("=" * 60)

# Try to get model from local variables (if running after notebook)
try:
    # Check if model exists in current scope
    if 'model' in dir():
        print("✓ Found 'model' in scope")
        model.save("saved_model/toxic_lstm.h5")
        print("✓ Model saved to saved_model/toxic_lstm.h5")
    else:
        print("❌ 'model' not found. Please run your training notebook first.")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Try to get tokenizer from local variables
try:
    if 'tokenizer' in dir():
        print("✓ Found 'tokenizer' in scope")
        with open("saved_model/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        print("✓ Tokenizer saved to saved_model/tokenizer.pkl")
    else:
        print("❌ 'tokenizer' not found. Please run your training notebook first.")
except Exception as e:
    print(f"❌ Error saving tokenizer: {e}")

print("\n" + "=" * 60)
print("Now you can run: streamlit run app.py")
print("=" * 60)
