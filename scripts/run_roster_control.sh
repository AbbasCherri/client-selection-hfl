#!/usr/bin/env bash
# Roster-construction control — the isolating ablation for the paper_full N-sweep.
#
# Rationale, the confound it closes, and what each outcome means are in
# configs/roster_control.yaml. Read that first.
#
# `proposed_hfl` is NOT recomputed: it exists in results/paper_full at identical
# N, seeds and rounds, and reusing it is what keeps the pairing valid.
#
# 4 N-values x 1 method x 10 seeds = 40 jobs, ~3-4 h on 12 vCPU.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]]; then
    HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$HOME/.hf_env" | head -1)
    export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN missing"; exit 1; }

LOG=results/roster_control.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== roster-construction control ====="

# The alias is what makes this an isolating ablation rather than a second
# sample of seed noise; if it stopped being read the run would still produce a
# clean-looking table. check_roster_control asserts the two builders diverge
# only when slots are slack, which is the regime the comparison lives in.
for chk in check_seed_alias check_roster_control check_roster_width check_collapse_guard; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

say "--- roster_control ---"
if python -m uavbench run_paper_sim --config configs/roster_control.yaml; then
    python scripts/gate_collapse.py results/roster_control \
        || say "  !! degenerate cells — see the gate output above"
else
    say "  !! roster_control FAILED"
fi

git add -A -- results || true
git diff --cached --quiet || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add roster control results $(date -Is)"

say "===== roster control complete ====="
