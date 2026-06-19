#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

has_sudo=false
if command -v sudo >/dev/null 2>&1; then
    if sudo -n true >/dev/null 2>&1; then
        has_sudo=true
    fi
fi

echo "=== Setting up HFL Simulation Environment on GCP VM ==="

# Update package lists and install Python system packages only when sudo is available.
if [ "$has_sudo" = true ]; then
    sudo apt-get update -y

    # Install Python3 venv if not present
    sudo apt-get install -y python3-venv python3-pip
else
    echo "sudo is not available; skipping apt-based system setup."
    echo "Python 3, pip, and venv must already be installed on this machine."
fi

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
.venv/bin/pip install torch torchvision pandas scikit-learn "huggingface-hub>=0.32.0" hf_xet matplotlib tifffile pillow

# Allow the download script to use a higher default download parallelism on GCP.
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=${HF_XET_NUM_CONCURRENT_RANGE_GETS:-32}
export HF_MAX_WORKERS=${HF_MAX_WORKERS:-32}

# Freeze requirements
echo "Freezing dependencies..."
.venv/bin/pip freeze > requirements.txt

# Run the download script
echo "Running the data download script..."
.venv/bin/python download_data.py

echo "=== Setup completed successfully! ==="
