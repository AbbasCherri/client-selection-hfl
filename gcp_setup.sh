#!/bin/bash
# gcp_setup.sh
# Sets up the Python virtual environment on the GCP VM.
# Installs all dependencies for the HFL simulation + UAV PSO benchmark sweep.
set -e

has_sudo=false
if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    has_sudo=true
fi

echo "=== HFL + UAV Benchmark Environment Setup ==="

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

# Bootstrap pip
if [ ! -x ".venv/bin/pip" ]; then
    .venv/bin/python -m ensurepip --upgrade
fi

echo "Upgrading pip / setuptools / wheel …"
.venv/bin/pip install --upgrade pip setuptools wheel

# Core scientific + benchmark stack
echo "Installing scientific stack …"
.venv/bin/pip install \
    "numpy>=1.24.0,<3.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.11.0" \
    "matplotlib>=3.7.0" \
    "pillow>=10.0.0" \
    "requests>=2.31.0" \
    "tqdm>=4.65.0" \
    "joblib>=1.3.0" \
    "pyarrow>=14.0.0" \
    "PyYAML>=6.0" \
    "filelock>=3.12.0" \
    "typing_extensions>=4.7.0"

# HuggingFace tooling – required for real dataset streaming
echo "Installing HuggingFace tooling …"
.venv/bin/pip install \
    "huggingface_hub>=0.32.0" \
    "datasets>=2.14.0" \
    hf_xet

# CPU-only PyTorch (avoids large CUDA runtimes on CPU-only VMs)
echo "Installing PyTorch (CPU) …"
.venv/bin/pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install the project packages in editable mode so both hflsim and uavbench
# are importable without setting PYTHONPATH manually.
echo "Installing project packages (editable) …"
.venv/bin/pip install -e ".[dev]"

# HF streaming performance tuning
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
export HF_MAX_WORKERS="${HF_MAX_WORKERS:-12}"
export HF_DATASET_REVISION="${HF_DATASET_REVISION:-6cf97c900445e080e61cb45e1aa72515d3ff1de8}"
export HF_HUB_DISABLE_UPDATE_CHECK=1

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source .venv/bin/activate"
echo "Run sweep with: HF_TOKEN=hf_xxx nohup ./run_gcp.sh & disown"
