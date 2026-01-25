@echo off
REM MoodPlay Local Development Setup Script
REM This script installs all dependencies and starts the services

echo ========================================
echo MoodPlay - Video Colorization Pipeline
echo Local Development Setup
echo ========================================
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Create virtual environment if not exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA (comment out if no GPU)
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create required directories
echo Creating directories...
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "checkpoints" mkdir checkpoints
if not exist "lora" mkdir lora
if not exist "configs" mkdir configs

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo   1. Start Redis (or use local mock)
echo   2. Run: python run.py
echo.
echo Or use run_local.bat for all-in-one start
echo.
pause
