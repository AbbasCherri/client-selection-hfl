#!/bin/bash
# run_gcp.sh — Self-terminating GCP wrapper for the Tier-2 scalability sweep.
#
# Runs the full (N=30..250) × (6 methods: pso/ga/centroid/random/static/no_uav) grid
# in parallel across all 8 vCPUs of the n1-standard-8 instance, then stops the VM.
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
PROJECT_ID="your-gcp-project-id"       # e.g. my-project-123
ZONE="us-central1-a"                   # zone where this VM lives
INSTANCE_NAME="$(hostname)"            # GCP VMs return their own name via hostname

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

# Pin top-level thread count — each joblib worker sets its own torch thread
# budget to 1, so the total across 8 workers stays at 8 (= vCPU count).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# HF token — required for real dataset streaming
# ---------------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "[$(date)] HF_TOKEN set — using real HFL dataset." | tee -a "$LOG_FILE"
else
    echo "[$(date)] WARNING: HF_TOKEN not set." | tee -a "$LOG_FILE"
    echo "[$(date)] Set HF_TOKEN=hf_xxx before launching. Falling back to synthetic sweep config." | tee -a "$LOG_FILE"
    SWEEP_CFG="configs/tier2_sweep_synthetic.yaml"
fi

SWEEP_CFG="${SWEEP_CFG:-configs/tier2_sweep.yaml}"

# ---------------------------------------------------------------------------
# Self-terminating shutdown — fires on success AND any error
# ---------------------------------------------------------------------------
shutdown_vm() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date)] Sweep finished successfully. Stopping VM." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ERROR: script exited with code $exit_code. Stopping VM anyway." | tee -a "$LOG_FILE"
    fi
    gcloud compute instances stop "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT_ID" \
        --quiet >> "$LOG_FILE" 2>&1 || true
}
trap shutdown_vm EXIT

# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
echo "[$(date)] ===== GCP sweep started ====="              | tee -a "$LOG_FILE"
echo "[$(date)] Project dir  : $SCRIPT_DIR"                 | tee -a "$LOG_FILE"
echo "[$(date)] Config       : $SWEEP_CFG"                  | tee -a "$LOG_FILE"
echo "[$(date)] Log          : $LOG_FILE"                   | tee -a "$LOG_FILE"
echo "[$(date)] Python       : $(python3 --version)"         | tee -a "$LOG_FILE"
echo "[$(date)] vCPUs        : $(nproc)"                    | tee -a "$LOG_FILE"
echo "[$(date)] RAM          : $(free -h | awk '/^Mem/{print $2}')" | tee -a "$LOG_FILE"
echo "[$(date)] Disk free    : $(df -h "$SCRIPT_DIR" | awk 'NR==2{print $4}')" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Run the sweep — N×method grid, 8-core parallel
# ---------------------------------------------------------------------------
echo "[$(date)] ----- Starting N-scalability sweep -----" | tee -a "$LOG_FILE"
echo "[$(date)] N: 30 50 100 150 200 250"                   | tee -a "$LOG_FILE"
echo "[$(date)] Methods: pso ga centroid random static no_uav" | tee -a "$LOG_FILE"
echo "[$(date)] 36 jobs total on 12 workers"                | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
uavbench run_sweep --config "$SWEEP_CFG" >> "$LOG_FILE" 2>&1

echo "[$(date)] Sweep done." | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Disk summary
# ---------------------------------------------------------------------------
echo "[$(date)] Results disk usage:" | tee -a "$LOG_FILE"
du -sh "$RESULTS_DIR"/* 2>/dev/null | tee -a "$LOG_FILE" || true
df -h "$SCRIPT_DIR" | tee -a "$LOG_FILE"

echo "[$(date)] ===== GCP sweep complete =====" | tee -a "$LOG_FILE"
