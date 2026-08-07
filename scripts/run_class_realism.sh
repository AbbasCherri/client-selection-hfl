#!/usr/bin/env bash
# Class-awareness realism ablation: four arms, paired, at N=200.
#
# Answers the question the 2026-08 rigor plan left open at item 0.5 — the
# proposed system uses ground-truth per-client label histograms for BOTH client
# selection and UAV placement, and every result to date was produced on that
# oracle. See configs/class_realism.yaml for the design and the pre-registered
# reading of each pairwise contrast; do not reinterpret after seeing numbers.
#
#   class_realism_true      true   + class-aware placement   (oracle)
#   class_realism_pseudo    pseudo + class-aware placement   (zero disclosure)
#   class_realism_noplace   true   + class-blind placement   (placement's share)
#   class_realism_none      none   + class-blind placement   (lower anchor)
#
# 4 arms x N{200} x 1 method x 10 seeds = 40 full HFL runs, ~6-8 h on 12 vCPU.
# Every job is checkpointed, so this is safe to interrupt and relaunch.
#
# DO NOT start this while another sweep is running — both size themselves to the
# whole machine and would halve each other's throughput.
#
# SELF-STOPS ON SUCCESS ONLY; a failure leaves the instance up for inspection.
#
# Usage:  nohup ./scripts/run_class_realism.sh &
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
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN missing — the real-data pipeline cannot run"; exit 1
fi

LOG=results/class_realism.log
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

say "===== Class-awareness realism ablation ====="

# The pseudo rung was unreachable from run_full_hfl until 2026-08-07 (it called
# build_class_info with no model and raised). These gates cover that path and
# the degrade-together invariant between the histogram and class-aware
# placement — a silently-blind arm would look exactly like a real result.
for chk in check_class_histograms check_integration_hfl; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

STATUS=0
for arm in class_realism class_realism_pseudo class_realism_noplace class_realism_none; do
    say "--- arm: ${arm} ---"
    python -m uavbench run_paper_sim --config "configs/${arm}.yaml" || STATUS=$?
    commit_block "${arm}"
done

if [[ $STATUS -ne 0 ]]; then
    say "===== FAILED (exit $STATUS) — VM left RUNNING ====="
    exit $STATUS
fi

say "--- merging arms for paired significance ---"
python scripts/merge_class_realism.py || STATUS=$?

# Every metric, not just macro-F1: reporting macro-F1 alone is what previously
# hid that the selection advantage does not transfer to accuracy.
for metric in macro_f1 accuracy f1_collapsed f1_missing f1_obstructed f1_survived; do
    say "significance vs the oracle arm: ${metric}"
    python -m uavbench significance --config results/class_realism \
        --metric "$metric" --reference true_placeaware --correction-scope group || true
done

commit_block "class-realism-significance"

if [[ $STATUS -ne 0 ]]; then
    say "===== analysis FAILED (exit $STATUS) — VM left RUNNING ====="
    exit $STATUS
fi

say "===== Class-realism ablation complete — stopping the VM ====="
sudo shutdown -h +2 "Class realism ablation complete"
