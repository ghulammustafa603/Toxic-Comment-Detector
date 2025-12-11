@echo off
REM Restart Streamlit app with cache cleared

echo.
echo ========================================
echo  Restarting Toxic Comment Detector
echo ========================================
echo.

REM Stop any running Streamlit instances
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

REM Clear Streamlit cache
if exist .streamlit (
    rmdir /s /q .streamlit 2>nul
    echo Cache cleared
)

echo Starting Streamlit application...
echo.
streamlit run app.py --server.headless true

pause

