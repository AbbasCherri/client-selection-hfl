#!/usr/bin/env bash
# C2 at higher power, plus the client-count generalisation it never had.
#
# READ FIRST: REPORTS/preregistration_c2_power.md, committed 0bb7abec before any
# of these seeds were computed. It fixes n=25 with NO optional stopping, the
# three criteria, and the fact that C2 was selected for extension after seeing
# v6 — which must be disclosed wherever this is reported.
#
# Six configs, two arms x three grids. The arms differ only in
# fl.uav_weight_mode, which is a config field rather than a method name, so they
# cannot share one config; identical method name and seed index means both draw
# identical seeds and the pairing is still exact.
#
#   grid (a) fleet:    K in {5,10,15,20,30} at N=200   125 jobs x 2 arms
#   grid (b) N=50:     K in {10,20}                     50 jobs x 2 arms
#   grid (b) N=100:    K in {10,20}                     50 jobs x 2 arms
#
# ~450 jobs, ~6.5 h on 12 vCPU. Checkpointed per (K, seed, method) — safe to
# interrupt and relaunch.
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

LOG=results/c2_power.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== C2 power + N-generalisation ====="

for chk in check_uav_weight_mode check_shard_diversity check_collapse_guard \
           check_altitude_band; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

FAILED=""
for stem in c2_power_base c2_power_c2 \
            c2_power_n50_base c2_power_n50_c2 \
            c2_power_n100_base c2_power_n100_c2; do
    say "--- ${stem} ---"
    if python -m uavbench run_uav_sweep --config "configs/${stem}.yaml"; then
        python scripts/gate_collapse.py "results/${stem}" \
            || say "  !! ${stem} has degenerate cells — see the gate output above"
    else
        say "  !! ${stem} FAILED"
        FAILED="$FAILED ${stem}"
    fi
    git add -A -- results || true
    git diff --cached --quiet || \
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add ${stem} results $(date -Is)"
done

if [[ -n "$FAILED" ]]; then
    say "ARMS FAILED:$FAILED — not scoring a partial grid"
    exit 1
fi

say "--- scoring against REPORTS/preregistration_c2_power.md §4 ---"
python scripts/score_c2_power.py | tee results/c2_power_verdict.txt
git add -A -- results || true
git diff --cached --quiet || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add C2 power verdict $(date -Is)"

say "===== C2 power run complete ====="
