@echo off
REM Quick start script for Car Manual Q&A Application (Windows)

echo Car Manual Q&A Application
echo ==============================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
if not exist "venv\.installed" (
    echo Installing dependencies...
    pip install -r requirements.txt
    type nul > venv\.installed
) else (
    echo Dependencies already installed.
)

REM Check if database exists
if not exist "chroma_db" (
    echo.
    echo Setting up database (first time only)...
    python setup_data.py
)

REM Run the application
echo.
echo Starting Streamlit application...
streamlit run app.py

@REM Made with Bob
