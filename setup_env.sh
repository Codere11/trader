#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install playwright browsers
playwright install

echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"