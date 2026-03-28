#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "venv/.setup_complete" ]; then
    echo "Installing dependencies..."
    python3 setup_env.py
fi

echo "Starting up application..."
python3 main_gui.py
