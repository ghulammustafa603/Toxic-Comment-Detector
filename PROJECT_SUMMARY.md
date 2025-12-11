# 🛡️ Toxic Comment Detector - Project Summary

## Quick Overview

**Project Name**: Toxic Comment Detector  
**Type**: Deep Learning Semester Project  
**Technology**: Bidirectional LSTM Neural Network  
**Interface**: Streamlit Web Application  
**Classification Type**: Multi-Label (6 categories)

---

## 🎯 What Does It Do?

Automatically detects and classifies toxic comments in 6 categories:
1. **Toxic** - General toxic behavior
2. **Severe Toxic** - Extreme toxicity
3. **Obscene** - Vulgar language
4. **Threat** - Threatening language
5. **Insult** - Insulting language
6. **Identity Hate** - Hate speech

---

## 🏗️ Architecture

```
Input Text → Embedding → Bidirectional LSTM → Dropout → Dense → Output (6 scores)
```

**Key Components**:
- **Embedding Layer**: 128 dimensions, 20K vocabulary
- **Bidirectional LSTM**: 64 units (processes both directions)
- **Dropout**: 0.5 and 0.3 (prevents overfitting)
- **Dense Layer**: 64 units with ReLU
- **Output**: 6 units with Sigmoid (multi-label)

---

## 📊 Dataset

- **Source**: Kaggle Toxic Comment Classification Challenge
- **Size**: ~160,000 labeled comments
- **Split**: 80% training, 20% testing
- **Labels**: 6 binary labels per comment

---

## ⚙️ Features

### 1. Single Comment Analysis
- Real-time toxicity detection
- Individual scores for all 6 categories
- Visual charts and metrics

### 2. Batch Processing
- Upload CSV files
- Analyze multiple comments
- Export results

### 3. Adjustable Threshold
- Fine-tune sensitivity (0.0 - 1.0)
- Balance false positives/negatives

### 4. User-Friendly Interface
- Modern web-based GUI
- Three tabs: Detect, Batch Analysis, Information
- Detailed visualizations

---

## 🔧 Technical Stack

- **Deep Learning**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **ML Tools**: scikit-learn
- **Language**: Python 3.x

---

## 📈 Performance

- **Training Accuracy**: ~99%+
- **Validation Accuracy**: ~99%+
- **Processing Speed**: Real-time (<1 second per comment)
- **Scalability**: Handles batch files efficiently

---

## 🚀 How to Use

### Installation
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python train_model.py
```

### Run Application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Toxic-Comment-Detector/
├── app.py              # Streamlit GUI
├── train_model.py      # Training script
├── data/               # Dataset
├── saved_model/        # Trained model
└── notebook/           # Development notebooks
```

---

## 🎓 Learning Outcomes

- Deep learning model development
- Text preprocessing and tokenization
- Multi-label classification
- Web application development
- End-to-end ML project implementation

---

## 🌟 Key Achievements

✅ High accuracy toxicity detection  
✅ Multi-label classification  
✅ User-friendly web interface  
✅ Real-time and batch processing  
✅ Complete, deployable system  

---

## 🔮 Future Enhancements

- Multi-language support
- Transfer learning (BERT/GPT)
- Model explainability
- API development
- Cloud deployment

---

## 📝 Use Cases

- Social media moderation
- Online forum management
- Comment section filtering
- Chat application safety
- Content moderation pipelines

---

**For detailed presentation content, see PRESENTATION_GUIDE.md**

