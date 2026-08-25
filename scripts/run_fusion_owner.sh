#!/usr/bin/env bash
# Fusion-ownership control — is the UAV TIER, not the selector, what loses?
#
# Pre-registration, arms, contrasts and decision rules are in
# configs/fusion_owner.yaml. Read that first; it was committed before this ran.
#
# proposed_hfl(uav) and flat_fl are NOT recomputed: both exist in
# results/paper_full at identical N, seeds and rounds, and reusing them is what
# keeps the two pairings exact.
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

LOG=results/fusion_owner.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== fusion-ownership control ====="

# check_fusion_owner is the one that matters: it asserts the knob is read (else
# the arm scores as a flat null while testing nothing) and that flat_fl does not
# move with it (else the C2 pairing against paper_full's flat_fl is invalid).
for chk in check_fusion_owner check_collapse_guard check_sweep_resume; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

say "--- fusion_owner (proposed_hfl, fusion_owner=client) ---"
if python -m uavbench run_paper_sim --config configs/fusion_owner.yaml; then
    python scripts/gate_collapse.py results/fusion_owner \
        || say "  !! degenerate cells — see the gate output above"
    say "--- scoring against the pre-registered criteria ---"
    python scripts/score_fusion_owner.py | tee results/fusion_owner_verdict.txt
else
    say "  !! fusion_owner FAILED"
fi

git add -A -- results || true
git diff --cached --quiet || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add fusion owner results $(date -Is)"

say "===== fusion-ownership control complete ====="
