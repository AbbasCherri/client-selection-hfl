#!/usr/bin/env bash
# VM #1 block 2: wait out Phase 6, then re-run 0.3 with the fair_mab fix.
#
# WHY 0.3 RUNS AGAIN. Its first pass (2026-08-03, 110 jobs) returned
# bit-identical means AND stds for all four fair_mab arms across 10 seeds. Two
# independent causes, both fixed in a91c30f1:
#   * the cap arms patched DEFAULT_T_STALE_CAP, a function default that both
#     call sites shadow with an explicit t_stale_cap= argument;
#   * at the 1x cap every client's staleness is pinned to 1.0 at every
#     selection event, so the reward ranking is battery order no matter what
#     the weights are.
# The arms now vary FAIRMAB_STALE_CAP_MULT (read at the call site) and the
# weight arms run at a cap where staleness actually discriminates.
#
# The six non-fair_mab arms are recomputed too, not out of caution but because
# c690c0d6 folds each arm's constant overrides into the resume signature —
# their stored checkpoints predate that key and are correctly judged stale.
# One code version across the whole block is better provenance anyway.
#
# Sequential with Phase 6 on purpose: both saturate all 12 vCPUs, and running
# them together would just make each slower and muddy the wall-clock numbers.
#
# Does NOT stop the VM. Usage:  sudo nohup ./scripts/run_vm1_block2.sh &
set -uo pipefail          # not -e: a failing block must not skip the commit
cd "$(dirname "$0")/.."

# Self-activate: this runs under sudo (results/ is root-owned), which does not
# inherit the caller's venv — the first Phase 6 launch died on exactly that.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG=results/vm1_block2.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -Is)] $*"; }

commit_block() {
    local label="$1"
    say "committing after $label"
    git add -A -- results || true
    if git diff --cached --quiet; then
        say "  nothing new to commit"
    else
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add $label results $(date -Is)" || true
        say "  committed $(git rev-parse --short HEAD)"
    fi
}

say "===== VM1 block 2 ====="

# Wait for Phase 6 rather than racing it. Poll the script itself, not a PID
# captured at launch, so this is safe to start at any point.
while pgrep -f "run_env_screen.sh" > /dev/null; do
    say "Phase 6 still running — waiting 5 min"
    sleep 300
done
say "Phase 6 not running; proceeding"
commit_block "env-screen"

say "0.3 re-run — baseline constants with the fair_mab fix (11 arms x N=500 x 10 seeds)"
python -m uavbench run_selection_sim --config configs/baseline_constants.yaml \
    || say "0.3 FAILED"
commit_block "baseline-constants-v2"

say "===== VM1 block 2 complete — VM left RUNNING ====="
