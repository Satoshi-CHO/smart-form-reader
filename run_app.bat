@echo off
cd /d %~dp0
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate

if not exist "venv\.setup_complete" (
    echo Installing dependencies...
    python setup_env.py
)

echo Starting up application...
python main_gui.py
if %errorlevel% neq 0 pause
