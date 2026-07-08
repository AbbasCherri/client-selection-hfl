#!/bin/bash
# run_selection_gcp.sh — Self-terminating GCP wrapper for the selection-isolation benchmark.
#
# Runs the client-selection rules head-to-head (5 modes: ucb/random/fedcs/
# rep_cap/fair_mab × 6 N-values (30..500) × 3 seeds = 90 jobs, 200 rounds,
# static elbow K-means UAVs, ~10% data subsample) in parallel across all
# 12 vCPUs, then stops the VM.
#
# Usage (SSH into the VM, then):
#   chmod +x run_selection_gcp.sh
#   HF_TOKEN=hf_xxx nohup ./run_selection_gcp.sh &
#   disown
#   # close SSH — VM stops itself when done
#
# Set SELF_STOP=0 to keep the VM running after the sweep (e.g. for debugging).
# The VM is STOPPED (not deleted); disk, results/, and the log are preserved.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — fill in before uploading to the VM
# ---------------------------------------------------------------------------
PROJECT_ID="project-bacf2da8-2fce-4137-a90"
ZONE="us-central1-a"
INSTANCE_NAME="instance-20260703-060853"

SELECTION_CFG="${SELECTION_CFG:-configs/selection_isolation.yaml}"
SELF_STOP="${SELF_STOP:-1}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/selection_sim.log"
RESULTS_DIR="$SCRIPT_DIR/results"

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    VENV="$SCRIPT_DIR/.venv"
elif [ -f "$(dirname "$SCRIPT_DIR")/.venv/bin/activate" ]; then
    VENV="$(dirname "$SCRIPT_DIR")/.venv"
else
    echo "[$(date)] ERROR: .venv not found. Run ./gcp_setup.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src"

# OMP/BLAS thread limits are set INSIDE the parallel FL worker processes via
# torch.set_num_threads(1) in the sweep code.  Setting them here globally would
# also throttle the single-threaded ResNet-18 feature pre-fetch to 1 thread.

# Throttle HuggingFace parallel fetchers during the sequential pre-fetch phase
# to avoid 429 rate limits. Workers never call HF — only the pre-fetch does.
export HF_MAX_WORKERS=2
export HF_HUB_DISABLE_UPDATE_CHECK=1
export HF_DATASET_REVISION="6cf97c900445e080e61cb45e1aa72515d3ff1de8"
export HF_HUB_DOWNLOAD_TIMEOUT=300

# ---------------------------------------------------------------------------
# HF token — required for real dataset streaming
# ---------------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "[$(date)] HF_TOKEN set — using real HFL dataset." | tee -a "$LOG_FILE"
else
    echo "[$(date)] ERROR: HF_TOKEN not set." | tee -a "$LOG_FILE"
    echo "[$(date)] selection_isolation.yaml requires the real dataset (data.source: real). Set HF_TOKEN=hf_xxx before launching." | tee -a "$LOG_FILE"
    exit 1
fi

# ---------------------------------------------------------------------------
# Self-terminating shutdown — fires on success AND any error
# ---------------------------------------------------------------------------
shutdown_vm() {
    local exit_code=$?
    if [ "$SELF_STOP" != "1" ]; then
        echo "[$(date)] SELF_STOP=0 — leaving VM running (exit code $exit_code)." | tee -a "$LOG_FILE"
        return
    fi
    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date)] Selection simulation finished successfully. Stopping VM." | tee -a "$LOG_FILE"
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
echo "[$(date)] ===== GCP selection-isolation simulation started =====" | tee -a "$LOG_FILE"
echo "[$(date)] Project dir  : $SCRIPT_DIR"                  | tee -a "$LOG_FILE"
echo "[$(date)] Config       : $SELECTION_CFG"               | tee -a "$LOG_FILE"
echo "[$(date)] Log          : $LOG_FILE"                    | tee -a "$LOG_FILE"
echo "[$(date)] Python       : $(python3 --version)"          | tee -a "$LOG_FILE"
echo "[$(date)] vCPUs        : $(nproc)"                     | tee -a "$LOG_FILE"
echo "[$(date)] RAM          : $(free -h | awk '/^Mem/{print $2}')" | tee -a "$LOG_FILE"
echo "[$(date)] Disk free    : $(df -h "$SCRIPT_DIR" | awk 'NR==2{print $4}')" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Run the selection-isolation sweep — mode×N×seed grid, 12-core parallel
# ---------------------------------------------------------------------------
echo "[$(date)] ----- Starting selection-isolation sweep -----" | tee -a "$LOG_FILE"
echo "[$(date)] Modes: ucb random fedcs rep_cap fair_mab | N: 30 50 100 200 350 500 | seeds: 3 — 90 jobs on 12 workers" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
python3 -m uavbench run_selection_sim --config "$SELECTION_CFG" >> "$LOG_FILE" 2>&1

echo "[$(date)] Selection-isolation sweep done." | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Disk summary
# ---------------------------------------------------------------------------
echo "[$(date)] Results disk usage:" | tee -a "$LOG_FILE"
du -sh "$RESULTS_DIR"/* 2>/dev/null | tee -a "$LOG_FILE" || true
df -h "$SCRIPT_DIR" | tee -a "$LOG_FILE"

echo "[$(date)] ===== GCP selection-isolation simulation complete =====" | tee -a "$LOG_FILE"
