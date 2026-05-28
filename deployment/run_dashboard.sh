#!/bin/bash
# Script to run the Streamlit dashboard

# Check if virtual environment exists, if not create it
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python -m venv ../venv
fi

# Activate virtual environment
source ../venv/bin/activate

# Install dependencies if needed
pip install -r ../requirements.txt

# Run the Streamlit dashboard
echo "Starting Enterprise LLM Trust Framework Dashboard..."
streamlit run ../src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0