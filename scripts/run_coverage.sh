#!/usr/bin/env bash
# The placement sweep, re-run over realistic radii with the new placement stack.
#
# What changed since the previous coverage sweep, and why its result does not
# carry over:
#   * `placement_fitness` was each optimizer's SELF-reported score, so
#     mozaffari/alzenad were measured at their own path-loss radius (618 m)
#     while everyone else was measured at R_comm. The bias flipped sign with
#     R_comm, making the column uninterpretable rather than merely shifted.
#   * coverage was a flat range gate, under which altitude only ever subtracts
#     usable radius — a 3D deployment whose vertical dimension was degenerate.
#     It is now the shared Al-Hourani air-to-ground channel, calibrated per job
#     so the best achievable radius equals the swept R_comm.
#   * the grid bottomed out at 2 km, where every method already covers most
#     clients. It now starts at 250 m, where placement actually binds.
# PIPELINE_VERSION 4 refuses the old checkpoints for exactly these reasons.
#
# SELF-STOPS ON SUCCESS ONLY — a failure leaves the instance up for inspection.
# Three earlier runners never stopped and burned ~37 h of idle VM between them.
#
# Usage:  nohup ./scripts/run_coverage.sh &
set -uo pipefail          # not -e: a failing block must not skip the commit
cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]]; then
    # Token lives outside the repo on purpose — the repo is public.
    HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$HOME/.hf_env" | head -1)
    export HF_TOKEN
fi

LOG=results/coverage.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -Is)] $*"; }

commit_block() {
    git add -A -- results || true
    if git diff --cached --quiet; then
        say "  nothing new to commit"
    else
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add $1 results $(date -Is)" || true
        say "  committed $(git rev-parse --short HEAD)"
    fi
}

say "===== Coverage sweep: realistic radii, path-loss link model ====="

# A 2-day sweep whose placement methods were silently mis-scored would look
# entirely normal in its output and would have to be discarded. These gates
# cover the equal-radius re-score, the 3D altitude requirement, the
# vertical/horizontal decoupling, and the MCLP bound.
for chk in check_placement_methods check_mclp; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

python -m uavbench run_coverage_sweep --config configs/paper_coverage.yaml
STATUS=$?
commit_block "coverage-sweep-v2"

if [[ $STATUS -ne 0 ]]; then
    say "===== Coverage sweep FAILED (exit $STATUS) — VM left RUNNING ====="
    exit $STATUS
fi

say "===== Coverage sweep complete — stopping the VM ====="
sudo shutdown -h +2 "Coverage sweep complete"
