#!/bin/bash
# run_paper_sim.sh — Full paper system simulation on GCP.
#
# Runs the complete §V comparison:
#   proposed_hfl | flat_fl | centralized | hfl_no_selection | hfl_static | hfl_no_reputation
# across N ∈ {100, 200, 500} × 3 seeds = 54 parallel jobs on 8 vCPUs.
#
# Estimated wall-clock: 2-3 hours on n1-standard-8.
#
# Usage (SSH into the VM, then):
#   chmod +x run_paper_sim.sh
#   HF_TOKEN=hf_xxx nohup ./run_paper_sim.sh &
#   disown
#   # SSH can be closed — VM stops itself when done
#
# Results land at:   results/paper_full/
# Log at:            paper_sim.log
#
# The VM is STOPPED (not deleted) when the run finishes or fails.
# Disk, results/, and paper_sim.log are preserved.

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
LOG_FILE="$SCRIPT_DIR/paper_sim.log"
RESULTS_DIR="$SCRIPT_DIR/results/paper_full"
SIM_CFG="configs/paper_full.yaml"

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

# Pin threads: each worker sets torch.set_num_threads(1) internally
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# HuggingFace rate-limit guard (only sequential pre-fetch phase touches HF)
export HF_MAX_WORKERS=2
export HF_HUB_DISABLE_UPDATE_CHECK=1
export HF_DATASET_REVISION="6cf97c900445e080e61cb45e1aa72515d3ff1de8"

# ---------------------------------------------------------------------------
# HF token check
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[$(date)] ERROR: HF_TOKEN is not set." | tee -a "$LOG_FILE"
    echo "[$(date)] Set HF_TOKEN=hf_xxx before launching." | tee -a "$LOG_FILE"
    exit 1
fi
export HF_TOKEN

# ---------------------------------------------------------------------------
# Self-terminating shutdown — fires on success AND any error
# ---------------------------------------------------------------------------
shutdown_vm() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date)] Simulation finished successfully. Stopping VM." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ERROR: script exited with code $exit_code. Stopping VM anyway." | tee -a "$LOG_FILE"
    fi
    if gcloud compute instances stop "$INSTANCE_NAME" \
           --zone="$ZONE" \
           --project="$PROJECT_ID" \
           --quiet >> "$LOG_FILE" 2>&1; then
        echo "[$(date)] gcloud stop issued." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] gcloud stop failed — using sudo shutdown." | tee -a "$LOG_FILE"
        sudo shutdown -h now >> "$LOG_FILE" 2>&1 || true
    fi
}
trap shutdown_vm EXIT

# ---------------------------------------------------------------------------
# Remove hf_xet if still installed (causes spurious tree walks)
# ---------------------------------------------------------------------------
cd "$SCRIPT_DIR"
if pip show hf_xet > /dev/null 2>&1; then
    echo "[$(date)] Uninstalling hf_xet …" | tee -a "$LOG_FILE"
    pip uninstall -y hf_xet >> "$LOG_FILE" 2>&1
fi

# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
echo "[$(date)] ===== Paper system simulation started =====" | tee -a "$LOG_FILE"
echo "[$(date)] Project dir  : $SCRIPT_DIR"                  | tee -a "$LOG_FILE"
echo "[$(date)] Config       : $SIM_CFG"                     | tee -a "$LOG_FILE"
echo "[$(date)] Log          : $LOG_FILE"                    | tee -a "$LOG_FILE"
echo "[$(date)] Python       : $(python3 --version)"          | tee -a "$LOG_FILE"
echo "[$(date)] vCPUs        : $(nproc)"                     | tee -a "$LOG_FILE"
echo "[$(date)] RAM          : $(free -h | awk '/^Mem/{print $2}')" | tee -a "$LOG_FILE"
echo "[$(date)] Disk free    : $(df -h "$SCRIPT_DIR" | awk 'NR==2{print $4}')" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Experiment: full paper system comparison
# 6 methods × 3 N-values × 3 seeds = 54 jobs
# ---------------------------------------------------------------------------
echo "[$(date)] ----- Full paper system simulation -----"                   | tee -a "$LOG_FILE"
echo "[$(date)] Methods: proposed_hfl flat_fl centralized hfl_no_selection hfl_static hfl_no_reputation" | tee -a "$LOG_FILE"
echo "[$(date)] N: 100 200 500  |  seeds: 3  |  54 jobs total on 8 workers" | tee -a "$LOG_FILE"

python3 -m uavbench run_paper_sim --config "$SIM_CFG" >> "$LOG_FILE" 2>&1

echo "[$(date)] Paper simulation done." | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Disk summary
# ---------------------------------------------------------------------------
echo "[$(date)] Results disk usage:" | tee -a "$LOG_FILE"
du -sh "$RESULTS_DIR"  2>/dev/null | tee -a "$LOG_FILE" || true
df -h "$SCRIPT_DIR"    | tee -a "$LOG_FILE"

echo "[$(date)] ===== Paper system simulation complete =====" | tee -a "$LOG_FILE"
