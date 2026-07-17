#!/bin/bash
# run_gcp.sh — Self-terminating GCP wrapper for the whole paper pipeline.
#
# Runs scripts/reproduce_paper.sh end to end (Tier-1 grid -> analyze/plot ->
# paper sim -> selection isolation -> N-sweep -> stress sweep -> significance
# -> artifact staging), commits + pushes results/ to GitHub, then stops the
# VM. This is the only GCP entry point —
# it replaces the old run_paper_sim.sh / run_selection_gcp.sh, which each ran
# a subset of what reproduce_paper.sh already does end to end.
#
# Every long-running step underneath (uavbench run / run_paper_sim /
# run_selection_sim / run_sweep / run_stress_sweep) checkpoints as it goes:
# each (N, method/mode, seed) job is skipped and reloaded from disk instead
# of recomputed if it already finished on a prior attempt. So if this script
# (or the VM) dies partway through, just re-launch it exactly as before —
# it resumes from the last completed job rather than starting over.
#
# Usage (SSH into the VM, then):
#   chmod +x run_gcp.sh
#   HF_TOKEN=hf_xxx ./run_gcp.sh
#   # prints "backgrounded as PID ..." then returns control to your shell —
#   # close the SSH session whenever; the script has already detached itself
#   # (setsid + nohup + disown) and the VM stops itself when the run finishes
#   # (or errors, or if killed and not relaunched — a preempted/killed run
#   # just sits paused, resumable)
#
# Set SMOKE=1 for the reduced end-to-end check instead of the full grid.
# Set SELF_STOP=0 to keep the VM running after the run (e.g. for debugging).
# The VM is STOPPED (not deleted); disk, results/, and the log are preserved.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — verify before uploading to the VM (instance may have changed
# since this was last run; `gcloud compute instances list` to check)
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-project-bacf2da8-2fce-4137-a90}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-instance-20260703-060853}"

SMOKE="${SMOKE:-0}"
SELF_STOP="${SELF_STOP:-1}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root (script lives in scripts/)
SELF="$SCRIPT_DIR/scripts/run_gcp.sh"
LOG_FILE="$SCRIPT_DIR/gcp_run.log"
VENV="$SCRIPT_DIR/.venv"

# ---------------------------------------------------------------------------
# Fast pre-flight checks — fail loud in THIS terminal, before detaching, so
# an obvious misconfiguration (no venv, no token) doesn't just silently sit
# in a background process's log until you go looking for it.
# ---------------------------------------------------------------------------
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: .venv not found. Run scripts/gcp_setup.sh first." >&2
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export HF_TOKEN=hf_xxx before launching." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-detach: on first invocation, re-exec this script under setsid+nohup
# with output redirected to $LOG_FILE, disown it from the shell's job table,
# then exit immediately. setsid gives the child its own session (no
# controlling terminal at all, so an SSH hangup can't reach it even before
# nohup's SIGHUP-ignoring kicks in); disown drops it from bash's job list so
# closing the shell doesn't wait on or warn about it. Everything from here
# down runs in that detached child (guarded by the env-var sentinel so the
# re-exec doesn't loop).
# ---------------------------------------------------------------------------
if [ "${_RUN_GCP_DAEMONIZED:-0}" != "1" ]; then
    export _RUN_GCP_DAEMONIZED=1
    setsid nohup "$SELF" </dev/null >>"$LOG_FILE" 2>&1 &
    bg_pid=$!
    disown
    echo "run_gcp.sh backgrounded as PID $bg_pid — detached from this terminal."
    echo "Log:  $LOG_FILE"
    echo "Tail: tail -f $LOG_FILE"
    echo "Safe to close this SSH session now; the VM stops itself when the run finishes (or errors)."
    exit 0
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_TOKEN

# OMP/BLAS thread limits are set INSIDE the parallel FL worker processes via
# torch.set_num_threads(1) in the sweep code.  Setting them here globally
# would also throttle the single-threaded ResNet-18 feature pre-fetch to 1
# thread, turning a 20-min feature extraction into a 3-hour one.  Do NOT
# export OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS here.

# Throttle HuggingFace parallel fetchers during the sequential pre-fetch phase
# to avoid 429 rate limits. Workers never call HF — only the pre-fetch does.
export HF_MAX_WORKERS=2
export HF_HUB_DISABLE_UPDATE_CHECK=1
export HF_DATASET_REVISION="${HF_DATASET_REVISION:-6cf97c900445e080e61cb45e1aa72515d3ff1de8}"
export HF_HUB_DOWNLOAD_TIMEOUT=300

# ---------------------------------------------------------------------------
# Remove hf_xet if still installed (causes spurious recursive tree walks)
# ---------------------------------------------------------------------------
cd "$SCRIPT_DIR"
if pip show hf_xet > /dev/null 2>&1; then
    echo "[$(date)] Uninstalling hf_xet …" | tee -a "$LOG_FILE"
    pip uninstall -y hf_xet >> "$LOG_FILE" 2>&1
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
        echo "[$(date)] Pipeline finished successfully. Stopping VM." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ERROR: script exited with code $exit_code. Stopping VM anyway (results so far are checkpointed — relaunch to resume instead of losing this run)." | tee -a "$LOG_FILE"
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
echo "[$(date)] ===== GCP pipeline run started (smoke=$SMOKE) =====" | tee -a "$LOG_FILE"
echo "[$(date)] Project dir  : $SCRIPT_DIR"                  | tee -a "$LOG_FILE"
echo "[$(date)] Log          : $LOG_FILE (reproduce_paper.sh logs its own steps to results/reproduce_paper.log)" | tee -a "$LOG_FILE"
echo "[$(date)] Python       : $(python3 --version)"          | tee -a "$LOG_FILE"
echo "[$(date)] vCPUs        : $(nproc)"                     | tee -a "$LOG_FILE"
echo "[$(date)] RAM          : $(free -h | awk '/^Mem/{print $2}')" | tee -a "$LOG_FILE"
echo "[$(date)] Disk free    : $(df -h "$SCRIPT_DIR" | awk 'NR==2{print $4}')" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Run the whole pipeline (checkpointed — safe to kill and relaunch)
# ---------------------------------------------------------------------------
echo "[$(date)] ----- Starting scripts/reproduce_paper.sh -----" | tee -a "$LOG_FILE"

if [[ "$SMOKE" == "1" ]]; then
    "$SCRIPT_DIR/scripts/reproduce_paper.sh" --smoke
else
    "$SCRIPT_DIR/scripts/reproduce_paper.sh"
fi

echo "[$(date)] Pipeline done." | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Disk summary
# ---------------------------------------------------------------------------
echo "[$(date)] Results disk usage:" | tee -a "$LOG_FILE"
du -sh "$SCRIPT_DIR/results"/* 2>/dev/null | tee -a "$LOG_FILE" || true
df -h "$SCRIPT_DIR" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Commit + push results to GitHub before the VM stops itself. Runs only after
# a successful pipeline (set -e aborts earlier otherwise). Feature caches
# (img_features.npy) and *.log are excluded via .gitignore, so nothing staged
# here can hit GitHub's 100 MB file limit. On any git failure the script
# exits non-zero: the shutdown trap still stops the VM, the results stay on
# its disk, and the log says exactly what to push manually.
# ---------------------------------------------------------------------------
echo "[$(date)] Committing results to GitHub …" | tee -a "$LOG_FILE"
git add -A -- results >> "$LOG_FILE" 2>&1
if git diff --cached --quiet; then
    echo "[$(date)] No new results to commit." | tee -a "$LOG_FILE"
elif git commit -m "Add results from GCP run $(date -Is) (smoke=$SMOKE)" >> "$LOG_FILE" 2>&1 \
    && git pull --rebase origin main >> "$LOG_FILE" 2>&1 \
    && git push origin main >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] Results pushed to GitHub: $(git rev-parse --short HEAD)" | tee -a "$LOG_FILE"
else
    git rebase --abort >> "$LOG_FILE" 2>&1 || true
    echo "[$(date)] ERROR: committing/pushing results FAILED — results remain on the VM disk; restart the VM and push manually." | tee -a "$LOG_FILE"
    exit 1
fi

echo "[$(date)] ===== GCP pipeline run complete =====" | tee -a "$LOG_FILE"
