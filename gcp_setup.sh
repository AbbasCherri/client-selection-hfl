#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Setting up HFL Simulation Environment on GCP VM ==="

# Update package lists
sudo apt-get update -y

# Install Python3 venv if not present
sudo apt-get install -y python3-venv python3-pip

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
else
    echo "Virtual environment .venv already exists."
fi

# Upgrade pip
echo "Upgrading pip..."
.venv/bin/pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies (PyTorch, torchvision, pandas, scikit-learn, etc.)..."
.venv/bin/pip install torch torchvision pandas scikit-learn huggingface-hub matplotlib tifffile pillow

# Freeze requirements
echo "Freezing dependencies..."
.venv/bin/pip freeze > requirements.txt

# Run the download script
echo "Running the data download script..."
.venv/bin/python download_data.py

echo "=== Setup completed successfully! ==="
