#!/usr/bin/env bash
# Phase 1 of the August 2026 rigor plan (REPORTS/rigor_plan_2026-08.md):
# the class-awareness / oracle-degradation experiment.
#
# This is the block that can change what the paper claims, so it runs BEFORE
# the expensive paper_full re-run. Pre-registered reading of the outcome is in
# the plan — do not reinterpret after seeing the numbers.
#
#   selection_isolation  10 arms x N{100,200,500} x 10 seeds  (300 jobs)
#   selection_dp_ladder   3 arms x N{500}         x 10 seeds  ( 30 jobs)
#
# ~30 h on a 12-vCPU VM. Every (N, arm, seed) job is checkpointed, so this is
# safe to interrupt and relaunch unchanged — it resumes where it stopped.
#
# Usage on the VM:  nohup ./scripts/run_phase1.sh > /tmp/phase1.log 2>&1 &
#
# STOP_VM=1 stops the instance when the run finishes (credit protection while
# unattended). The trap fires on ANY exit, so if you need to kill this script,
# SIGKILL it first or the VM stops under you — same footgun as run_gcp.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

STOP_VM="${STOP_VM:-0}"
LOG=results/phase1.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

stop_vm_if_asked() {
    local rc=$?
    echo "[$(date -Is)] exiting with status $rc"
    if [[ "$STOP_VM" == "1" ]]; then
        echo "[$(date -Is)] stopping instance to conserve credits"
        sudo shutdown -h now || true
    fi
}
trap stop_vm_if_asked EXIT

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN missing — the real-data pipeline cannot run"; exit 1
fi

step() { echo; echo "--- [$(date -Is)] $* ---"; }

step "Phase 1a: selection isolation (10 arms x 3 N x 10 seeds)"
python -m uavbench run_selection_sim --config configs/selection_isolation.yaml

step "Phase 1b: DP epsilon ladder (3 arms x N=500 x 10 seeds)"
python -m uavbench run_selection_sim --config configs/selection_dp_ladder.yaml

# Significance on BOTH metrics and every per-class F1. Reporting only macro-F1
# is what hid the finding that the advantage does not transfer to accuracy.
for metric in macro_f1 accuracy f1_collapsed f1_missing f1_obstructed f1_survived; do
    step "Significance vs ucb: $metric"
    python -m uavbench significance --config results/selection_isolation \
        --metric "$metric" --reference ucb --correction-scope group || true
    python -m uavbench significance --config results/selection_dp_ladder \
        --metric "$metric" --reference ucb_dp_eps1 --correction-scope group || true
done

step "Disk usage"
du -sh results/selection_isolation results/selection_dp_ladder 2>/dev/null || true

step "Committing results"
git add -A -- results
if git diff --cached --quiet; then
    echo "no new results to commit"
else
    git commit -m "Add Phase 1 class-awareness results $(date -Is)"
    echo "committed $(git rev-parse --short HEAD) — pull via git bundle (VM cannot push)"
fi

echo "[$(date -Is)] ===== Phase 1 complete ====="
