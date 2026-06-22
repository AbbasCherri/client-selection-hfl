#!/bin/bash
# gcp_setup.sh
# Sets up the Python virtual environment on a GCP VM (or any Linux host).
# Installs all dependencies needed for streaming-based HFL simulation.
set -e

has_sudo=false
if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    has_sudo=true
fi

echo "=== HFL Simulation Environment Setup ==="

if [ "$has_sudo" = true ]; then
    sudo apt-get update -y
    sudo apt-get install -y python3-venv python3-pip
else
    echo "sudo unavailable – skipping apt setup; python3/venv must already be installed."
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv …"
    python3 -m venv .venv
else
    echo "Virtual environment .venv already exists."
fi

# Bootstrap pip if needed
if [ ! -x ".venv/bin/pip" ]; then
    .venv/bin/python -m ensurepip --upgrade
fi

echo "Upgrading pip / setuptools / wheel …"
.venv/bin/pip install --upgrade pip setuptools wheel

# Core scientific stack
echo "Installing scientific stack …"
.venv/bin/pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    pillow \
    requests \
    tqdm

# HuggingFace tooling – 'datasets' is required for streaming mode
echo "Installing HuggingFace tooling …"
.venv/bin/pip install \
    "huggingface-hub>=0.32.0" \
    "datasets>=2.14.0" \
    hf_xet

# CPU-only PyTorch (avoids pulling large CUDA runtimes on CPU-only VMs)
echo "Installing PyTorch (CPU) …"
.venv/bin/pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Environment variables for faster HF streaming
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
export HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
export HF_DATASET_REVISION="${HF_DATASET_REVISION:-6cf97c900445e080e61cb45e1aa72515d3ff1de8}"
export HF_HUB_DISABLE_UPDATE_CHECK=1

# Freeze requirements
echo "Freezing requirements …"
.venv/bin/pip freeze > requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:  source .venv/bin/activate"
echo "Run simulations with:           bash run_all_simulations.sh"
