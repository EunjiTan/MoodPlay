@echo off
REM MoodPlay Local Run Script
REM Runs all services locally without Docker

echo ========================================
echo MoodPlay - Starting Local Services
echo ========================================
echo.

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found.
    echo Run setup_local.bat first!
    pause
    exit /b 1
)

REM Check if Redis is running (optional - we'll use memory broker if not)
echo Checking Redis...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo Redis not available - using in-memory task queue
    set USE_MEMORY_BROKER=1
) else (
    echo Redis is running
    set USE_MEMORY_BROKER=0
)

REM Start Celery worker in background (optional for async processing)
echo Starting Celery workers...
start "MoodPlay SAM Worker" cmd /c "celery -A backend.workers.tasks_seg.celery_app worker --loglevel=info -Q sam3_queue -P solo"
start "MoodPlay Gen Worker" cmd /c "celery -A backend.workers.tasks_gen.celery_app worker --loglevel=info -Q generation_queue -P solo"

REM Wait for workers to start
timeout /t 3 /nobreak >nul

REM Start Flask app
echo.
echo Starting Flask API server...
echo.
echo ========================================
echo MoodPlay running at: http://localhost:5000
echo Press Ctrl+C to stop
echo ========================================
echo.

python run.py
