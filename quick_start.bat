@echo off
REM Quick Start Script for Brain Tumor Classification Project
REM This script helps you get started quickly

echo ============================================================
echo Brain Tumor Classification - Quick Start
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version
echo.

echo [2/4] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Try running: pip install -r requirements.txt manually
    pause
    exit /b 1
)
echo.

echo [3/4] Setting up project directories...
python setup_project.py
if errorlevel 1 (
    echo.
    echo ERROR: Failed to setup project
    pause
    exit /b 1
)
echo.

echo [4/4] Setup complete!
echo.
echo ============================================================
echo Next Steps:
echo ============================================================
echo.
echo 1. Add your training images to the Training folder
echo    - Training\glioma\
echo    - Training\meningioma\
echo    - Training\no_tumor\
echo    - Training\pituitary\
echo.
echo 2. Add your testing images to the Testing folder
echo    (same structure as Training)
echo.
echo 3. Train the model:
echo    python train.py
echo.
echo 4. Make predictions:
echo    - Command line: python predict.py
echo    - Web app: streamlit run app.py
echo.
echo ============================================================
echo.
pause
