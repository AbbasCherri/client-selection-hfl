#!/usr/bin/env bash
# Final VM #1 block of the August 2026 rigor plan: 0.3 + Phase 6.
#
#   0.3  baseline_constants  — the literature baselines' own knobs at settings
#        other than the ones we hand-picked (~6 h). Half of the tuning
#        asymmetry: our selector got 120 Optuna trials, they got none.
#   6    env_screen          — one-at-a-time screening of the simulator's
#        environment constants, T_MAX_S first (~6 h).
#
# COMMITS AFTER EVERY BLOCK. VM #1 is near the end of its credits; if they run
# out mid-run the instance becomes unreachable, and anything not yet committed
# is stranded on a disk we cannot get back to. Committing per block bounds that
# loss to the block in flight. (Results still have to be pulled by bundle from
# the laptop — the VM has no GitHub credential.)
#
# Does NOT stop the VM: the last thing that killed a run was a trap firing on a
# path nobody intended. Stopping is left to the operator.
set -uo pipefail          # not -e: a failing block must not skip the commit
cd "$(dirname "$0")/.."

LOG=results/vm1_tail.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -Is)] $*"; }

if [[ -z "${HF_TOKEN:-}" ]]; then say "HF_TOKEN missing — aborting"; exit 1; fi

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

say "===== VM1 tail: 0.3 baseline constants, then Phase 6 env screen ====="

say "0.3 — baseline-constant provenance sweep (11 arms x N=500 x 10 seeds)"
python -m uavbench run_selection_sim --config configs/baseline_constants.yaml \
    || say "0.3 FAILED (continuing to Phase 6)"
commit_block "baseline-constants"

say "Phase 6 — environment-constant screening (12 cells x 3 arms x 5 seeds)"
./scripts/run_env_screen.sh || say "Phase 6 FAILED"
commit_block "env-screen"

say "===== VM1 tail complete — VM left RUNNING, stop it manually ====="
