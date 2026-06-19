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

# Bootstrap pip if the venv was created without it.
if [ ! -x ".venv/bin/pip" ]; then
    echo "Bootstrapping pip in .venv..."
    .venv/bin/python -m ensurepip --upgrade
fi

# Upgrade pip
echo "Upgrading pip..."
.venv/bin/pip install --upgrade pip setuptools wheel

# Install project dependencies required by the Python imports in this repo.
echo "Installing Python dependencies (NumPy, pandas, PyTorch, torchvision, scikit-learn, matplotlib, Pillow, and Hugging Face tooling)..."
.venv/bin/pip install \
    numpy \
    pandas \
    torch \
    torchvision \
    scikit-learn \
    matplotlib \
    pillow \
    "huggingface-hub>=0.32.0" \
    hf_xet

# Allow the download script to use a higher default download parallelism on GCP.
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}
export HF_MAX_WORKERS=${HF_MAX_WORKERS:-4}
export HF_DATASET_REVISION=${HF_DATASET_REVISION:-6cf97c900445e080e61cb45e1aa72515d3ff1de8}
export HF_HUB_DISABLE_UPDATE_CHECK=1

# Freeze requirements
echo "Freezing requirements..."
.venv/bin/pip freeze > requirements.txt

echo "=== Setup completed successfully! ==="
