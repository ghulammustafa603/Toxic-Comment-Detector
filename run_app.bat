@echo off
REM Toxic Comment Detector - Streamlit Launcher
REM This script starts the Streamlit application

echo.
echo ========================================
echo  Toxic Comment Detector - Streamlit GUI
echo ========================================
echo.

REM Check if model exists
if not exist "saved_model\toxic_lstm.h5" (
    echo WARNING: Model not found!
    echo.
    echo Please run the training script first:
    echo   python train_model.py
    echo.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

echo Starting Streamlit application...
echo.
streamlit run app.py
