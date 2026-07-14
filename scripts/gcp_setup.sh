#!/bin/bash
# gcp_setup.sh
# Sets up the Python virtual environment on the GCP VM.
# Installs all dependencies for the HFL simulation + UAV PSO benchmark sweep.
#
# Dependency versions live in requirements.txt (single source of truth,
# mirrors pyproject.toml) — do not add inline pins here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# CPU-only PyTorch FIRST, from the CPU wheel index — so the generic
# torch/torchvision pins in requirements.txt resolve as already satisfied
# instead of pulling multi-GB CUDA runtimes onto a CPU-only VM.
echo "Installing PyTorch (CPU) …"
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Everything else — versions pinned in requirements.txt.
echo "Installing project dependencies from requirements.txt …"
.venv/bin/pip install -r requirements.txt

# hf_xet does a full recursive tree walk of the dataset on every access,
# causing 429/504 errors on large repos. Our loader uses streaming=True
# which bypasses xet entirely — remove it if already installed.
.venv/bin/pip uninstall -y hf_xet 2>/dev/null || true

# Install the project packages in editable mode so both hflsim and uavbench
# are importable without setting PYTHONPATH manually.
echo "Installing project packages (editable) …"
.venv/bin/pip install -e ".[dev]"

# Sanity check: both packages import and the CLI resolves.
echo "Verifying installation …"
.venv/bin/python -c "import hflsim, uavbench, torch, sklearn, datasets; print('  imports OK — torch', torch.__version__)"
.venv/bin/python -m uavbench --help >/dev/null && echo "  uavbench CLI OK"

echo ""
echo "=== Setup complete ==="
echo "Activate with:        source .venv/bin/activate"
echo "Paper sweep:          HF_TOKEN=hf_xxx nohup scripts/run_gcp.sh & disown"
echo "Selection isolation:  HF_TOKEN=hf_xxx nohup scripts/run_selection_gcp.sh & disown"
echo "(HF runtime tuning — HF_MAX_WORKERS, HF_HUB_DOWNLOAD_TIMEOUT, dataset"
echo " revision pin — is exported by the run scripts, not here: exports from"
echo " this setup script would not survive into your shell anyway.)"
