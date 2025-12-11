# 📋 Quick Reference Card - Toxic Comment Detector

## 🎯 Project in 30 Seconds

**What**: Deep learning system that detects toxic comments in 6 categories  
**How**: Bidirectional LSTM neural network  
**Interface**: Streamlit web application  
**Accuracy**: ~99%+  
**Speed**: Real-time (<1 second per comment)

---

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| Dataset Size | ~160,000 comments |
| Toxicity Categories | 6 types |
| Model Accuracy | ~99%+ |
| Vocabulary Size | 20,000 words |
| Sequence Length | 200 tokens |
| Training Epochs | 5 |
| Batch Size | 128 |
| Processing Speed | <1 second/comment |

---

## 🏗️ Architecture (One Line)

**Input → Embedding (128D) → BiLSTM (64) → Dropout → Dense (64) → Output (6)**

---

## 🎯 6 Toxicity Categories

1. **Toxic** - General toxic behavior
2. **Severe Toxic** - Extreme toxicity  
3. **Obscene** - Vulgar language
4. **Threat** - Threatening language
5. **Insult** - Insulting language
6. **Identity Hate** - Hate speech

---

## ⚡ Key Features

- ✅ Single comment analysis (real-time)
- ✅ Batch CSV processing
- ✅ Adjustable threshold (0.0-1.0)
- ✅ Detailed visualizations
- ✅ Export results

---

## 🔧 Tech Stack

- **DL**: TensorFlow/Keras
- **Web**: Streamlit
- **Data**: Pandas, NumPy
- **ML**: scikit-learn
- **Language**: Python 3.x

---

## 📁 Project Files

- `app.py` - Streamlit GUI (422 lines)
- `train_model.py` - Training script (153 lines)
- `data/train.csv` - Dataset
- `saved_model/` - Trained model

---

## 🚀 Commands

```bash
# Install
pip install -r requirements.txt

# Train
python train_model.py

# Run
streamlit run app.py
```

---

## 💡 Key Points for Presentation

1. **Problem**: Millions of comments need automated moderation
2. **Solution**: Bidirectional LSTM for context-aware detection
3. **Innovation**: Multi-label classification (6 categories simultaneously)
4. **Result**: 99%+ accuracy, real-time processing
5. **Impact**: Deployable solution for online platforms

---

## 🎓 Learning Outcomes

- Deep learning model development
- Text preprocessing & tokenization
- Multi-label classification
- Web app development
- End-to-end ML project

---

## 📚 Documentation Files

- `PRESENTATION_GUIDE.md` - Complete presentation content
- `PPT_OUTLINE.md` - Slide-by-slide outline
- `PROJECT_SUMMARY.md` - Detailed summary
- `README.md` - User guide
- `QUICK_REFERENCE.md` - This file

---

**For detailed information, see PRESENTATION_GUIDE.md**

