# 🛡️ Toxic Comment Detector - Streamlit GUI

A comprehensive web-based application for detecting toxic comments using a Bidirectional LSTM neural network trained on multi-label toxicity classification.

## Features

- **Single Comment Analysis**: Analyze individual comments for toxicity
- **Batch Processing**: Upload CSV files and analyze multiple comments at once
- **Multi-Label Classification**: Detects 6 types of toxicity:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate
- **Adjustable Threshold**: Fine-tune sensitivity with a threshold slider
- **Detailed Visualization**: View confidence scores and detailed analysis
- **Export Results**: Download batch analysis results as CSV

## Project Structure

```
Toxic-Comment-Detector/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── saved_model/           # Pre-trained model and tokenizer
│   ├── toxic_lstm.h5      # Trained model
│   ├── tokenizer.pkl      # Tokenizer for text preprocessing
│   └── tokenizer_info.pkl # Tokenizer configuration
├── data/
│   └── train.csv          # Training dataset
└── notebook/
    └── comment-detector.ipynb  # Jupyter notebook
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your training data is located at:
```
data/train.csv
```

The CSV should have these columns:
- `comment_text`: The comment text
- `toxic`: Label (0 or 1)
- `severe_toxic`: Label (0 or 1)
- `obscene`: Label (0 or 1)
- `threat`: Label (0 or 1)
- `insult`: Label (0 or 1)
- `identity_hate`: Label (0 or 1)

### 3. Train the Model

Run the training script to train and save the model:

```bash
python train_model.py
```

This will:
- Load and preprocess the training data
- Train the LSTM model
- Save the model to `saved_model/toxic_lstm.h5`
- Save the tokenizer to `saved_model/tokenizer.pkl`

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

### Single Comment Analysis

1. Go to the "🔍 Detect" tab
2. Enter or paste a comment in the text area
3. Click "🔍 Analyze" button
4. View the toxicity scores for each category
5. Adjust the threshold slider to see how it affects detection

### Batch Analysis

1. Go to the "📊 Batch Analysis" tab
2. Upload a CSV file containing comments
3. Click "🔍 Analyze All Comments"
4. Wait for analysis to complete
5. View statistics and detailed results
6. Download results as CSV

### Model Information

1. Go to the "ℹ️ Information" tab
2. Learn about the model architecture and toxicity categories
3. Understand how preprocessing and predictions work

## Model Architecture

```
Input Text (200 tokens)
    ↓
Embedding Layer (128 dimensions)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.5)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (6 units, Sigmoid) → 6 toxicity predictions
```

## Text Preprocessing

The model preprocesses text by:
1. Converting to lowercase
2. Removing URLs (http/www patterns)
3. Removing special characters (keeping only letters and spaces)
4. Removing extra whitespace

## Threshold Settings

- **Lower threshold (e.g., 0.3)**: More sensitive, catches more toxic comments but with more false positives
- **Default threshold (0.5)**: Balanced sensitivity and specificity
- **Higher threshold (e.g., 0.7)**: Less sensitive, fewer false positives but may miss some toxic comments

## Performance Metrics

The model was trained on the Kaggle Toxic Comment Classification dataset containing ~160K comments with multiple toxicity labels. The bidirectional LSTM architecture allows the model to understand context from both directions, improving accuracy.

## File Descriptions

### app.py
Main Streamlit application with three tabs:
- **🔍 Detect**: Single comment analysis interface
- **📊 Batch Analysis**: Batch processing and CSV upload
- **ℹ️ Information**: Model documentation and architecture

### train_model.py
Training script that:
- Loads and preprocesses the training data
- Creates and trains the LSTM model
- Saves the trained model and tokenizer
- Displays training metrics

### requirements.txt
All Python package dependencies for the project

## Dependencies

- **streamlit**: Web framework for the GUI
- **tensorflow**: Deep learning framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **matplotlib/seaborn**: Data visualization

## Troubleshooting

### Model not found error
- Make sure you've run `python train_model.py` first
- Check that `saved_model/toxic_lstm.h5` exists

### Data loading error
- Verify the training data is at `data/train.csv`
- Ensure the CSV has the required columns

### GPU/Memory issues
- Reduce batch size in `train_model.py`
- Reduce number of training epochs

## Future Improvements

- Add model retraining capability through the GUI
- Implement model versioning and comparison
- Add real-time performance monitoring
- Support for additional languages
- Fine-tuning on domain-specific datasets
- API endpoint for external integration

## License

This project is for educational purposes.

## Author

Created as a semester project for Deep Learning course.

---

**Enjoy detecting toxic comments! 🛡️**
