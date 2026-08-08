#!/usr/bin/env bash
# Placement contrast at a BINDING radius — the correction to arms 1/3.
#
# Arms 1-4 (scripts/run_class_realism.sh) run at R_comm = 20 km, where
# proposed_hfl already covers 99.95% of clients. Class-aware placement steers
# *which* clients fall inside coverage, so at 99.95% it has nothing to steer and
# the 20 km placement contrast is null by construction — no power, no
# information. These two arms rerun that contrast at R_comm = 8 km, where
# coverage is partial and placement actually decides who participates.
#
# The 20 km arms remain valid for the SELECTION contrasts (oracle vs pseudo,
# and class-aware selection vs none): 120 UAV slots against 200 clients means
# selection binds regardless of coverage.
#
# 2 arms x N{200} x 1 method x 10 seeds = 20 jobs, ~1 h on 12 vCPU.
# Run AFTER run_class_realism.sh finishes — do not run both at once.
#
# SELF-STOPS ON SUCCESS ONLY.
#
# Usage:  nohup ./scripts/run_class_realism_bind.sh &
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

LOG=results/class_realism_bind.log
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

say "===== Class-realism: placement contrast at R_comm = 8 km ====="

STATUS=0
for arm in class_realism_bind class_realism_bind_noplace; do
    say "--- arm: ${arm} ---"
    python -m uavbench run_paper_sim --config "configs/${arm}.yaml" || STATUS=$?
    commit_block "${arm}"
done

if [[ $STATUS -ne 0 ]]; then
    say "===== FAILED (exit $STATUS) — VM left RUNNING ====="
    exit $STATUS
fi

say "--- merging the binding-radius pair ---"
python scripts/merge_class_realism.py --group bind || STATUS=$?

for metric in macro_f1 accuracy f1_collapsed f1_missing f1_obstructed f1_survived; do
    say "significance, class-aware vs class-blind placement: ${metric}"
    python -m uavbench significance --config results/class_realism_bind_merged \
        --metric "$metric" --reference bind_placeaware --correction-scope group || true
done

commit_block "class-realism-bind-significance"

if [[ $STATUS -ne 0 ]]; then
    say "===== analysis FAILED (exit $STATUS) — VM left RUNNING ====="
    exit $STATUS
fi

say "===== Binding-radius placement contrast complete — stopping the VM ====="
sudo shutdown -h +2 "Class realism bind complete"
