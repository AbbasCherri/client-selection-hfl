#!/bin/bash
# run_gcp.sh — Self-terminating GCP wrapper for the full paper system simulation.
#
# Runs the full paper comparison (6 methods: proposed_hfl/flat_fl/centralized/
# hfl_no_selection/hfl_static/hfl_no_reputation × 3 N-values × 3 seeds = 54 jobs)
# in parallel across all 8 vCPUs of the instance, then stops the VM.
#
# Usage (SSH into the VM, then):
#   chmod +x run_gcp.sh
#   HF_TOKEN=hf_xxx nohup ./run_gcp.sh &
#   disown
#   # close SSH — VM stops itself when done
#
# The VM is STOPPED (not deleted) when the run finishes or fails.
# Disk, results/, and simulation.log are preserved.
# To delete instead: replace `instances stop` with `instances delete --quiet`.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — fill in before uploading to the VM
# ---------------------------------------------------------------------------
PROJECT_ID="project-bacf2da8-2fce-4137-a90"
ZONE="us-central1-a"
INSTANCE_NAME="instance-20260615-095517"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/simulation.log"
RESULTS_DIR="$SCRIPT_DIR/results"

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    VENV="$SCRIPT_DIR/.venv"
elif [ -f "$(dirname "$SCRIPT_DIR")/.venv/bin/activate" ]; then
    VENV="$(dirname "$SCRIPT_DIR")/.venv"
else
    echo "[$(date)] ERROR: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src"

# Pin PyTorch/BLAS thread count per worker (each worker sets torch.set_num_threads(1)).
# Total active threads = 8 workers × 1 = 8 = vCPU count.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Throttle HuggingFace parallel fetchers during the sequential pre-fetch phase
# to avoid 429 rate limits. Workers never call HF — only the pre-fetch does.
export HF_MAX_WORKERS=2
export HF_HUB_DISABLE_UPDATE_CHECK=1
export HF_DATASET_REVISION="6cf97c900445e080e61cb45e1aa72515d3ff1de8"

# ---------------------------------------------------------------------------
# HF token — required for real dataset streaming
# ---------------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "[$(date)] HF_TOKEN set — using real HFL dataset." | tee -a "$LOG_FILE"
else
    echo "[$(date)] ERROR: HF_TOKEN not set." | tee -a "$LOG_FILE"
    echo "[$(date)] paper_full.yaml requires the real dataset (data.source: real). Set HF_TOKEN=hf_xxx before launching." | tee -a "$LOG_FILE"
    exit 1
fi

PAPER_CFG="${PAPER_CFG:-configs/paper_full.yaml}"

# ---------------------------------------------------------------------------
# Self-terminating shutdown — fires on success AND any error
# ---------------------------------------------------------------------------
shutdown_vm() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date)] Paper simulation finished successfully. Stopping VM." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ERROR: script exited with code $exit_code. Stopping VM anyway." | tee -a "$LOG_FILE"
    fi
    # Attempt self-stop via gcloud (requires the VM to have the 'compute' API scope).
    # If it fails (insufficient scope), fall back to `sudo shutdown` which always works.
    if gcloud compute instances stop "$INSTANCE_NAME" \
           --zone="$ZONE" \
           --project="$PROJECT_ID" \
           --quiet >> "$LOG_FILE" 2>&1; then
        echo "[$(date)] gcloud stop issued." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] gcloud stop failed (scope issue) — using sudo shutdown instead." | tee -a "$LOG_FILE"
        sudo shutdown -h now >> "$LOG_FILE" 2>&1 || true
    fi
}
trap shutdown_vm EXIT

# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
echo "[$(date)] ===== GCP paper simulation started ====="    | tee -a "$LOG_FILE"
echo "[$(date)] Project dir  : $SCRIPT_DIR"                 | tee -a "$LOG_FILE"
echo "[$(date)] Config       : $PAPER_CFG"                  | tee -a "$LOG_FILE"
echo "[$(date)] Log          : $LOG_FILE"                   | tee -a "$LOG_FILE"
echo "[$(date)] Python       : $(python3 --version)"         | tee -a "$LOG_FILE"
echo "[$(date)] vCPUs        : $(nproc)"                    | tee -a "$LOG_FILE"
echo "[$(date)] RAM          : $(free -h | awk '/^Mem/{print $2}')" | tee -a "$LOG_FILE"
echo "[$(date)] Disk free    : $(df -h "$SCRIPT_DIR" | awk 'NR==2{print $4}')" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Run the full paper system simulation — method×N×seed grid, 8-core parallel
# ---------------------------------------------------------------------------
echo "[$(date)] ----- Starting full paper simulation -----" | tee -a "$LOG_FILE"
echo "[$(date)] Methods: proposed_hfl flat_fl centralized hfl_no_selection hfl_static hfl_no_reputation" | tee -a "$LOG_FILE"
echo "[$(date)] N: 100 200 500, seeds: 3 — 54 jobs total on 8 workers" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
python3 -m uavbench run_paper_sim --config "$PAPER_CFG" >> "$LOG_FILE" 2>&1

echo "[$(date)] Paper simulation done." | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Disk summary
# ---------------------------------------------------------------------------
echo "[$(date)] Results disk usage:" | tee -a "$LOG_FILE"
du -sh "$RESULTS_DIR"/* 2>/dev/null | tee -a "$LOG_FILE" || true
df -h "$SCRIPT_DIR" | tee -a "$LOG_FILE"

echo "[$(date)] ===== GCP paper simulation complete =====" | tee -a "$LOG_FILE"
