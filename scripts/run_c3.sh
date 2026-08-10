#!/usr/bin/env bash
# C3 — the confirmatory arm, then the pre-registered verdict.
#
# READ FIRST: REPORTS/preregistration_v6_c3.md — §4 for the criteria (committed
# c02625c9, before any C3 code) and §5a for the exact operationalisation
# (committed 099fda45, after the screen but before this arm's code).
#
# One arm, deliberately. A C2+C3 combination is NOT the confirmatory test: C2 was
# chosen after seeing v6, so pairing it with C3 is exploratory by construction
# and §5a says it must be labelled that way if it is ever run.
#
# Baselines are reused from results/paper_uav_count at identical N/seeds/rounds
# rather than recomputed — that is what keeps the Wilcoxon pairing valid and it
# saves ~14 h.
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

LOG=results/c3.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== C3: redundancy-discounted placement coverage ====="

# The objective is duplicated across Fitness's scalar and batch paths; if only
# one honours it, the optimizer searches one thing and the table reports another.
for chk in check_disjoint_coverage check_coverage_mode check_collapse_guard \
           check_altitude_band; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

say "--- v6_c3_disjoint ---"
if python -m uavbench run_uav_sweep --config configs/v6_c3_disjoint.yaml; then
    python scripts/gate_collapse.py results/v6_c3_disjoint \
        || say "  !! degenerate cells — see the gate output above"
else
    say "  !! C3 arm FAILED"
fi
git add -A -- results || true
git diff --cached --quiet || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add C3 results $(date -Is)"

say "--- scoring against REPORTS/preregistration_v6_c3.md §4 ---"
python scripts/score_c3.py | tee results/c3_verdict.txt
git add -A -- results || true
git diff --cached --quiet || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add C3 verdict $(date -Is)"

say "===== C3 complete ====="
