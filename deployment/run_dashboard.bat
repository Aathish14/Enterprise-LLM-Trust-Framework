@echo off
REM Batch file to run the Streamlit dashboard on Windows

REM Check if virtual environment exists, if not create it
if not exist "..\venv" (
    echo Creating virtual environment...
    python -m venv ..\venv
)

REM Activate virtual environment
call ..\venv\Scripts\activate

REM Install dependencies if needed
pip install -r ..\requirements.txt

REM Run the Streamlit dashboard
echo Starting Enterprise LLM Trust Framework Dashboard...
streamlit run ..\src\dashboard\app.py --server.port 8501 --server.address 0.0.0.0