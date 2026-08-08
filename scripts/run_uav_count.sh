#!/usr/bin/env bash
# Fleet-size sweep: does placement matter more with fewer UAVs?
#
# The coverage sweep answered "does placement quality move macro-F1?" with a
# clear no — but it did so at a FIXED K=20, which covers 99.95% of clients at
# 20 km. That is the saturated regime, where no placement rule can distinguish
# itself. This sweep varies the fleet instead, down to K=2, where each aircraft's
# position decides who trains at all. If the spread across placement methods is
# still flat at K=2, the negative result is as strong as this benchmark can make
# it; if it opens up, placement matters and the coverage sweep was measuring the
# wrong axis.
#
# Capacity is swept with K to hold K*C = 120 — see configs/paper_uav_count.yaml
# for why, and for the caveat that small-K per-UAV capacities are not
# operationally realistic.
#
# 6 K-values x 17 methods x 10 seeds = 1020 jobs, ~10-12 h on 12 vCPU.
# Checkpointed per (K, seed, method) — safe to interrupt and relaunch.
#
# SELF-STOPS ON SUCCESS ONLY.
#
# Usage:  nohup ./scripts/run_uav_count.sh &
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
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN missing — the real-data pipeline cannot run"; exit 1
fi

LOG=results/uav_count.log
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

say "===== Fleet-size sweep: placement importance vs K ====="

# A 12 h sweep whose K never reached the jobs would look completely normal in
# its output and be worthless. These gates cover exactly that, plus the
# constant-slots invariant the design depends on.
for chk in check_uav_sweep check_placement_methods; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

python -m uavbench run_uav_sweep --config configs/paper_uav_count.yaml
STATUS=$?
commit_block "uav-count-sweep"

if [[ $STATUS -ne 0 ]]; then
    say "===== FAILED (exit $STATUS) — VM left RUNNING ====="
    exit $STATUS
fi

# Same reference as the coverage sweep so the two are read side by side.
for metric in macro_f1 coverage_pct accuracy; do
    say "significance vs mclp_place: ${metric}"
    python -m uavbench significance --config results/paper_uav_count \
        --metric "$metric" --reference mclp_place --correction-scope group || true
done

commit_block "uav-count-significance"

say "===== Fleet-size sweep complete — stopping the VM ====="
sudo shutdown -h +2 "UAV count sweep complete"
