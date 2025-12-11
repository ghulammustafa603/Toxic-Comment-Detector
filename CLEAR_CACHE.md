# 🗑️ How to Clear Streamlit Cache

## Quick Methods to Clear Cache

---

## Method 1: Using Streamlit Command (Easiest)

Open Command Prompt or Terminal and run:

```bash
streamlit cache clear
```

This will clear all cached data.

---

## Method 2: Manual Cache Deletion

### Windows:
```bash
# Navigate to your project folder
cd "E:\Deep Learning\semester project\Toxic-Comment-Detector"

# Delete cache folder
rmdir /s /q .streamlit 2>nul
```

### Mac/Linux:
```bash
# Navigate to your project folder
cd "path/to/Toxic-Comment-Detector"

# Delete cache folder
rm -rf .streamlit
```

---

## Method 3: Using PowerShell (Windows)

```powershell
# Navigate to project folder
cd "E:\Deep Learning\semester project\Toxic-Comment-Detector"

# Remove cache
Remove-Item -Path .streamlit -Recurse -Force -ErrorAction SilentlyContinue
```

---

## Method 4: Stop and Restart App

1. **Stop the app**: Press `Ctrl + C` in the terminal
2. **Wait 2-3 seconds**
3. **Restart**: `streamlit run app.py`

This will automatically clear the cache when you restart.

---

## Method 5: Use the Restart Script

I've created a `restart_app.bat` file that does this automatically:

1. **Double-click** `restart_app.bat` in your project folder
2. It will:
   - Stop any running Streamlit instances
   - Clear the cache
   - Restart the app

---

## Method 6: Clear Browser Cache (If Needed)

Sometimes the browser also caches the page:

1. **Chrome/Edge**: Press `Ctrl + Shift + Delete`
2. **Firefox**: Press `Ctrl + Shift + Delete`
3. **Or**: Press `Ctrl + F5` for hard refresh

---

## Complete Reset (Nuclear Option)

If nothing else works:

```bash
# Stop Streamlit
# Press Ctrl + C in terminal

# Clear Streamlit cache
streamlit cache clear

# Clear Python cache
# Delete __pycache__ folders
# Windows:
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

# Mac/Linux:
find . -type d -name __pycache__ -exec rm -r {} +

# Restart app
streamlit run app.py
```

---

## Quick Reference

**Fastest way:**
```bash
streamlit cache clear
streamlit run app.py
```

**Or just restart:**
1. `Ctrl + C` (stop)
2. `streamlit run app.py` (restart)

---

## Verify Cache is Cleared

After clearing cache, when you restart the app, you should see:
- "Loading model..." spinner (first time)
- Fresh model and tokenizer loading
- No cached errors

---

**That's it! Your cache is cleared.** 🎉

