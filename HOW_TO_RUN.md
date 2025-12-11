# 🚀 How to Run Toxic Comment Detector

## Quick Start Guide

---

## 📋 Prerequisites

- **Python 3.7 or higher** installed on your system
- **pip** (Python package installer)
- **Internet connection** (for first-time installation)

---

## 🎯 Option 1: Quick Run (If Model Already Exists)

If you already have the trained model in `saved_model/toxic_lstm.h5`, follow these steps:

### Step 1: Open Terminal/Command Prompt
- **Windows**: Press `Win + R`, type `cmd`, press Enter
- **Mac/Linux**: Open Terminal

### Step 2: Navigate to Project Folder
```bash
cd "E:\Deep Learning\semester project\Toxic-Comment-Detector"
```

### Step 3: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

**OR** (Windows only) - Double-click `run_app.bat`

### Step 5: Open in Browser
The app will automatically open in your browser at:
```
http://localhost:8501
```

---

## 🎯 Option 2: Complete Setup (First Time)

If you need to train the model first, follow these steps:

### Step 1: Install Python Dependencies
```bash
cd "E:\Deep Learning\semester project\Toxic-Comment-Detector"
pip install -r requirements.txt
```

This installs:
- TensorFlow (for deep learning)
- Streamlit (for web interface)
- Pandas, NumPy (for data processing)
- scikit-learn (for ML utilities)

### Step 2: Check Data File
Make sure `data/train.csv` exists with the required columns:
- `comment_text`
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

### Step 3: Train the Model
```bash
python train_model.py
```

**Note**: This will take some time (30 minutes to several hours depending on your computer)
- The script will:
  - Load and preprocess data
  - Train the LSTM model (5 epochs)
  - Save model to `saved_model/toxic_lstm.h5`
  - Save tokenizer to `saved_model/tokenizer.pkl`

### Step 4: Run the Application
```bash
streamlit run app.py
```

---

## 🖥️ Windows Quick Method

### Using the Batch File (Easiest)

1. **Double-click** `run_app.bat` in the project folder
2. The script will:
   - Check if model exists
   - Install dependencies if needed
   - Launch the Streamlit app
3. Browser will open automatically

---

## 📝 Step-by-Step Commands

### Complete Command Sequence:

```bash
# 1. Navigate to project folder
cd "E:\Deep Learning\semester project\Toxic-Comment-Detector"

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Train model (if not already trained)
python train_model.py

# 4. Run the application
streamlit run app.py
```

---

## ✅ Verification Checklist

Before running, check:

- [ ] Python is installed (`python --version`)
- [ ] You're in the correct folder
- [ ] `data/train.csv` exists (for training)
- [ ] `saved_model/toxic_lstm.h5` exists (for running app)
- [ ] `saved_model/tokenizer.pkl` exists (for running app)

---

## 🌐 Using the Application

Once the app opens in your browser:

### Tab 1: 🔍 Detect
1. Enter a comment in the text area
2. Click "🔍 Analyze"
3. View toxicity scores for all 6 categories
4. Adjust threshold slider if needed

### Tab 2: 📊 Batch Analysis
1. Upload a CSV file with comments
2. Click "🔍 Analyze All Comments"
3. View statistics and detailed results
4. Download results as CSV

### Tab 3: ℹ️ Information
- View model architecture details
- Learn about toxicity categories
- Understand how the system works

---

## ⚠️ Troubleshooting

### Problem: "Model not found"
**Solution**: 
```bash
python train_model.py
```

### Problem: "Module not found" or "No module named 'streamlit'"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Problem: "Port 8501 already in use"
**Solution**: 
- Close other Streamlit apps
- Or use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Problem: "CUDA/GPU errors" (if using GPU)
**Solution**: 
- Install TensorFlow CPU version:
```bash
pip install tensorflow-cpu
```

### Problem: Browser doesn't open automatically
**Solution**: 
- Manually open browser and go to: `http://localhost:8501`

### Problem: "Out of memory" during training
**Solution**: 
- Reduce batch size in `train_model.py` (change `batch_size=128` to `batch_size=64`)

---

## 🛑 Stopping the Application

- **In Terminal**: Press `Ctrl + C`
- **Close Browser**: Close the browser tab
- The server will stop automatically

---

## 📊 System Requirements

### Minimum:
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **Python**: 3.7+

### Recommended:
- **RAM**: 8 GB or more
- **Storage**: 5 GB free space
- **Python**: 3.8+
- **GPU**: Optional (for faster training)

---

## 🔄 Quick Reference Commands

```bash
# Check Python version
python --version

# Check if packages are installed
pip list | findstr streamlit
pip list | findstr tensorflow

# Check if model exists
dir saved_model\toxic_lstm.h5

# Run app (Windows)
run_app.bat

# Run app (Mac/Linux)
streamlit run app.py
```

---

## 💡 Tips

1. **First Run**: Training takes time, be patient
2. **Subsequent Runs**: If model exists, app starts in seconds
3. **Browser**: Use Chrome, Firefox, or Edge for best experience
4. **Performance**: Close other applications for faster training
5. **Updates**: Keep dependencies updated:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

---

## 📞 Need Help?

If you encounter issues:
1. Check the error message in terminal
2. Verify all files are in correct locations
3. Ensure Python and pip are working
4. Check README.md for more details

---

## 🎯 Summary

**Quickest way to run** (if model exists):
```bash
streamlit run app.py
```

**Complete setup** (first time):
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

**Windows users**:
- Just double-click `run_app.bat`!

---

**Happy detecting! 🛡️**

