#!/usr/bin/env bash
# v6 method evaluation — the 2x2 of C1 (reachable placement) and C2 (diversity-
# weighted edge aggregation), judged against criteria fixed in advance.
#
# READ FIRST: REPORTS/preregistration_v6_method.md, committed 9729a1ce before
# any v6 code existed. It names the comparators, the endpoint, the four
# conditions for declaring an improvement, and what is not allowed (no
# re-tuning, no post-hoc operating point, no reporting a subset of K).
#
# Order matters. The CONTROL runs first and the script stops if it fails: the
# C1/C2 edits touched greedy_assignment, AssignmentResult and both of Fitness's
# scoring paths, so if both switches OFF no longer reproduces v5's mclp_place
# numbers, then every arm after it is measuring the refactor rather than the
# method, and running them would waste four hours producing a result that
# cannot be interpreted.
#
# Baselines are NOT recomputed — moon2022 and mclp_place already exist in
# results/paper_uav_count at identical N, seeds and rounds. Reusing them is what
# makes the Wilcoxon pairing valid, and it saves ~14 h.
#
# ~200 jobs total, ~4-5 h on 12 vCPU. Run only when the VM is otherwise idle.
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

LOG=results/v6.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== v6 method evaluation ====="

for chk in check_coverage_mode check_uav_weight_mode check_shard_diversity \
           check_collapse_guard check_roster_width check_altitude_band; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

FAILED=""
run_arm() {                       # run_arm <config-stem>
    local stem="$1"
    local rdir="results/${stem}"
    say "--- ${stem} ---"
    if python -m uavbench run_uav_sweep --config "configs/${stem}.yaml"; then
        if ! python scripts/gate_collapse.py "$rdir"; then
            say "  !! ${stem} has degenerate cells — see the gate output above"
            FAILED="$FAILED ${stem}(collapsed)"
        fi
    else
        say "  !! ${stem} FAILED (exit $?)"
        FAILED="$FAILED ${stem}"
    fi
    git add -A -- results || true
    git diff --cached --quiet || \
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add ${stem} results $(date -Is)"
}

# ---- 0. control: both switches off must reproduce v5 --------------------
run_arm v6_control
if ! python scripts/check_v6_control.py; then
    say "CONTROL DID NOT REPRODUCE v5 — the C1/C2 edits changed the baseline."
    say "Every arm after this would measure the refactor, not the method."
    say "Aborting. Diff results/v6_control against results/paper_uav_count K20."
    exit 1
fi

# ---- 1. the 2x2 ----------------------------------------------------------
for arm in v6_c1_reachable v6_c2_diversity v6_both; do
    run_arm "$arm"
done

# ---- 2. judgement against the pre-registered criteria --------------------
say "--- scoring against REPORTS/preregistration_v6_method.md ---"
python scripts/score_v6.py | tee results/v6_verdict.txt
git add -A -- results || true
git diff --cached --quiet || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add v6 verdict $(date -Is)"

say "===== v6 evaluation complete ====="
[[ -z "$FAILED" ]] || { say "ARMS WITH PROBLEMS:$FAILED"; exit 1; }
