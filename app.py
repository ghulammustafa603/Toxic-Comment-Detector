import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1rem;
        }
        .toxic-high {
            color: #ff4444;
            font-weight: bold;
        }
        .toxic-medium {
            color: #ff9944;
            font-weight: bold;
        }
        .toxic-low {
            color: #44aa44;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Try multiple possible paths for the model
        possible_model_paths = [
            "saved_model/toxic_lstm.h5",
            "saved model/toxic_lstm.h5",
            "../saved_model/toxic_lstm.h5",
            "./saved_model/toxic_lstm.h5"
        ]
        
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            st.error("❌ Model file not found. Checked locations:")
            for path in possible_model_paths:
                st.error(f"   - {os.path.abspath(path)}")
            return None, None
        
        model = tf.keras.models.load_model(model_path)
        st.success(f"✓ Model loaded from: {model_path}")
        
        # Try to load tokenizer from multiple paths
        possible_tokenizer_paths = [
            "saved_model/tokenizer.pkl",
            "saved model/tokenizer.pkl",
            "../saved_model/tokenizer.pkl",
            "./saved_model/tokenizer.pkl"
        ]
        
        tokenizer = None
        for path in possible_tokenizer_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    tokenizer = pickle.load(f)
                break
        
        if tokenizer is None:
            st.warning("⚠️ Tokenizer not found. Model may not work correctly.")
            return model, None
        
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text)
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)  # only alphabets
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

def predict_toxicity(text, model, tokenizer):
    """Predict toxicity of given text"""
    if model is None or tokenizer is None:
        return None
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Tokenize and pad
    max_len = 200
    max_words = 20000
    
    text_seq = tokenizer.texts_to_sequences([cleaned_text])
    text_pad = pad_sequences(text_seq, maxlen=max_len)
    
    # Predict
    predictions = model.predict(text_pad, verbose=0)
    
    return predictions[0]

def get_toxicity_label(score):
    """Get label based on toxicity score"""
    if score > 0.7:
        return "🔴 High Toxicity"
    elif score > 0.4:
        return "🟠 Medium Toxicity"
    else:
        return "🟢 Low Toxicity"

# Main app
st.title("🛡️ Toxic Comment Detector")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses a **Bidirectional LSTM** neural network
    trained on toxic comments to detect multiple types of toxic behavior:
    
    - **Toxic**: General toxic behavior
    - **Severe Toxic**: Extreme toxic behavior
    - **Obscene**: Obscene language
    - **Threat**: Threatening language
    - **Insult**: Insulting language
    - **Identity Hate**: Hate speech
    """)
    
    st.markdown("---")
    st.header("Settings")
    threshold = st.slider(
        "Toxicity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for flagging toxicity"
    )

# Main content area
tabs = st.tabs(["🔍 Detect", "📊 Batch Analysis", "ℹ️ Information"])

# Tab 1: Single Comment Detection
with tabs[0]:
    st.header("Single Comment Analysis")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    if model is not None and tokenizer is not None:
        # Input area
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter a comment to analyze:",
                placeholder="Type or paste a comment here...",
                height=120,
                key="user_comment"
            )
        
        with col2:
            st.write("")
            st.write("")
            analyze_button = st.button("🔍 Analyze", use_container_width=True)
        
        if analyze_button and user_input.strip():
            # Get predictions
            predictions = predict_toxicity(user_input, model, tokenizer)
            
            if predictions is not None:
                # Display overall toxicity
                overall_toxicity = np.max(predictions)
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Overall Toxicity Score",
                        f"{overall_toxicity:.2%}",
                        delta=get_toxicity_label(overall_toxicity)
                    )
                
                with col2:
                    st.metric(
                        "Is Toxic?",
                        "YES" if overall_toxicity > threshold else "NO",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Threshold",
                        f"{threshold:.0%}"
                    )
                
                # Display detailed predictions
                st.markdown("### Detailed Analysis")
                
                labels = [
                    "Toxic",
                    "Severe Toxic",
                    "Obscene",
                    "Threat",
                    "Insult",
                    "Identity Hate"
                ]
                
                # Create columns for detailed scores
                cols = st.columns(3)
                for idx, label in enumerate(labels):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        score = predictions[idx]
                        color = "🔴" if score > threshold else "🟢"
                        st.metric(
                            f"{color} {label}",
                            f"{score:.2%}"
                        )
                
                # Visualize predictions
                st.markdown("### Score Distribution")
                chart_data = pd.DataFrame({
                    'Category': labels,
                    'Confidence': predictions,
                    'Threshold': [threshold] * len(labels)
                })
                
                st.bar_chart(chart_data.set_index('Category')[['Confidence']])
                
                # Display cleaned text
                with st.expander("View Cleaned Text"):
                    cleaned = clean_text(user_input)
                    st.code(cleaned, language="text")
    else:
        st.error("⚠️ Model not loaded. Please ensure the model file exists at 'saved_model/toxic_lstm.h5'")
        st.info("You can train the model using the training script provided in the project.")

# Tab 2: Batch Analysis
with tabs[1]:
    st.header("Batch Analysis")
    
    model, tokenizer = load_model_and_tokenizer()
    
    if model is not None and tokenizer is not None:
        uploaded_file = st.file_uploader(
            "Upload a CSV file with comments",
            type=["csv"],
            help="CSV should have a 'comment' or 'text' column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Find the text column
                text_col = None
                for col in ['comment', 'text', 'comment_text', 'content']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    st.error("CSV must contain one of these columns: comment, text, comment_text, or content")
                else:
                    if st.button("🔍 Analyze All Comments"):
                        # Analyze all comments
                        progress_bar = st.progress(0)
                        predictions_list = []
                        
                        for idx, comment in enumerate(df[text_col]):
                            pred = predict_toxicity(comment, model, tokenizer)
                            if pred is not None:
                                predictions_list.append(pred)
                            progress_bar.progress((idx + 1) / len(df))
                        
                        # Add predictions to dataframe
                        labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
                        for idx, label in enumerate(labels):
                            df[label] = [pred[idx] for pred in predictions_list]
                        
                        df['Max Toxicity'] = df[labels].max(axis=1)
                        df['Is Toxic'] = df['Max Toxicity'] > threshold
                        
                        # Display statistics
                        st.markdown("### Analysis Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Comments", len(df))
                        
                        with col2:
                            toxic_count = (df['Is Toxic']).sum()
                            st.metric("Toxic Comments", toxic_count)
                        
                        with col3:
                            st.metric("Clean Comments", len(df) - toxic_count)
                        
                        with col4:
                            toxicity_rate = (toxic_count / len(df)) * 100
                            st.metric("Toxicity Rate", f"{toxicity_rate:.1f}%")
                        
                        # Display dataframe
                        st.markdown("### Detailed Results")
                        display_df = df[[text_col] + labels + ['Max Toxicity', 'Is Toxic']].copy()
                        display_df['Max Toxicity'] = display_df['Max Toxicity'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="toxicity_analysis_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        st.error("⚠️ Model not loaded. Please ensure the model file exists.")

# Tab 3: Information
with tabs[2]:
    st.header("Model Information")
    
    st.markdown("""
    ### Model Architecture
    
    The model uses a **Bidirectional LSTM (Long Short-Term Memory)** neural network:
    
    1. **Embedding Layer**: Converts text to dense vectors (128 dimensions)
    2. **Bidirectional LSTM**: Processes text in both directions for better context understanding (64 units)
    3. **Dropout**: Prevents overfitting (50% dropout rate)
    4. **Dense Layer**: Feature extraction (64 units, ReLU activation)
    5. **Output Layer**: Multi-label classification (6 units, Sigmoid activation)
    
    ### Toxicity Categories
    
    - **Toxic**: General toxic behavior
    - **Severe Toxic**: Extreme levels of toxicity
    - **Obscene**: Obscene or vulgar language
    - **Threat**: Threatening or intimidating language
    - **Insult**: Insulting or demeaning language
    - **Identity Hate**: Hate speech targeting identity groups
    
    ### How It Works
    
    1. **Text Preprocessing**: The input text is cleaned by:
       - Converting to lowercase
       - Removing URLs
       - Removing special characters (keeping only letters and spaces)
       - Trimming extra whitespace
    
    2. **Tokenization**: Text is converted to sequences of integer tokens
    
    3. **Padding**: Sequences are padded to a fixed length (200 tokens)
    
    4. **Prediction**: The model predicts probabilities for each toxicity category
    
    ### Threshold
    
    The threshold slider allows you to adjust sensitivity:
    - **Lower threshold**: More sensitive, catches more toxic comments but with more false positives
    - **Higher threshold**: Less sensitive, fewer false positives but may miss some toxic comments
    
    ### Dataset
    
    The model was trained on the Kaggle Toxic Comment Classification Challenge dataset,
    which contains ~160K labeled comments with multiple toxicity categories.
    """)
    
    st.markdown("---")
    
    # Model details if loaded
    model, _ = load_model_and_tokenizer()
    if model is not None:
        st.markdown("### Model Summary")
        st.code(str(model.summary()), language="text")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.85rem;">
    Toxic Comment Detector | Built with Streamlit & TensorFlow | 2024
</div>
""", unsafe_allow_html=True)
