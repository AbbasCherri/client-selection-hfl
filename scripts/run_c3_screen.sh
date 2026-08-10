#!/usr/bin/env bash
# C3 screening — two one-round sweeps, then the pre-registered verdict.
#
# READ FIRST: REPORTS/preregistration_v6_c3.md, committed c02625c9 before any
# C3 code existed. It names both hypotheses and states, in advance, what kills
# each. This script only runs the screen; scripts/score_c3_screen.py applies the
# conditions.
#
# Cheap on purpose: 2 K x 2 methods x 10 seeds at ONE round, plus the same grid
# for the w=(1,0,0) arm. Minutes. The point is to spend a full fleet grid only on
# a hypothesis that has already survived something that could have killed it.
#
# No collapse gate here, deliberately: one round of training tells you nothing
# about whether a run learns, so gating on it would be theatre. Nothing from
# these results may be quoted except geometry and coverage columns.
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

LOG=results/c3_screen.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== C3 screening ====="

say "checking the geometry columns and the weights override are sound"
python tests/sanity_checks/check_placement_geometry.py || {
    say "check_placement_geometry FAILED — aborting before spending anything"
    exit 1
}

FAILED=""
for stem in c3_screen c3_screen_w100; do
    say "--- ${stem} ---"
    python -m uavbench run_uav_sweep --config "configs/${stem}.yaml" \
        || { say "  !! ${stem} FAILED"; FAILED="$FAILED ${stem}"; }
    git add -A -- results || true
    git diff --cached --quiet || \
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add ${stem} results $(date -Is)"
done

if [[ -n "$FAILED" ]]; then
    say "ARMS FAILED:$FAILED — not scoring a partial screen"
    exit 1
fi

say "--- scoring against the pre-registered falsification conditions ---"
python scripts/score_c3_screen.py | tee results/c3_screen_verdict.txt
git add -A -- results || true
git diff --cached --quiet || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add C3 screen verdict $(date -Is)"

say "===== C3 screening complete ====="
