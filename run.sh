#!/bin/bash

# Quick start script for Car Manual Q&A Application

echo "🚗 Car Manual Q&A Application"
echo "=============================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch venv/.installed
        echo "✓ Dependencies installed successfully!"
    else
        echo "✗ Error installing dependencies. Please check the error messages above."
        exit 1
    fi
else
    echo "Dependencies already installed."
fi

# Check if database exists
if [ ! -d "chroma_db" ]; then
    echo ""
    echo "Setting up database (first time only)..."
    python setup_data.py
    if [ $? -ne 0 ]; then
        echo "✗ Error setting up database. Please check the error messages above."
        exit 1
    fi
fi

# Run the application
echo ""
echo "Starting Streamlit application..."
streamlit run app.py
