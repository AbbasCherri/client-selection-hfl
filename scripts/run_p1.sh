#!/usr/bin/env bash
# P1 — re-tune every compared method AT THE OPERATING POINT (R_comm = 5 km).
#
# PREREGISTRATION.md §3: this is a FAIRNESS PREREQUISITE, not a hypothesis. It
# consumes no alpha because it tests nothing. It exists because every recipe in
# configs/tuned_weights.yaml was searched at R_comm = 20 km (results/hpo,
# 2026-08-05) and is applied at 5 km, so every arm in the paper runs
# hyperparameters fitted to a different physical regime.
#
# Symmetric by construction: EVERY compared method is re-tuned, not just the
# proposed one. Tuning one arm and imposing its recipe on the others is the
# defect that produced the inflated v4 result; doing it in the other direction
# would be the same error.
#
# flat_fl is INCLUDED, and it matters. It currently has no recipe of its own —
# it inherits proposed_hfl's — so it is the one arm that is under-tuned relative
# to the rest, and it is beating proposed_hfl anyway. Tuning it can only make
# the negative result stronger, which is exactly why it must be tuned.
#
# Objective is val_macro_f1 on tuning seeds 20-22, disjoint from evaluation
# seeds 0-19. Never reads the test metric.
#
# 50 + 3x30 = 140 trials, ~3 h on 12 vCPU after the feature warmup.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]]; then
    HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$HOME/.hf_env" | head -1); export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN missing"; exit 1; }

LOG=results/p1_retune.log
mkdir -p results results/hpo_5km
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

# The regime being tuned FOR. This is the whole point of the run.
R_COMM=5000
K_UAVS=20
CAPACITY=6

CACHE=results/hpo_cache_5km
STORAGE="sqlite:///$(pwd)/results/hpo_5km/weights.db"
SEEDS="20 21 22"
NVALS="30 50"
SUB=0.2
ROUNDS=20
PAR=6

commit_block() {
    git add -A -- results scripts PREREGISTRATION.md 2>/dev/null || true
    git diff --cached --quiet 2>/dev/null || \
        git -c user.email=vm@local -c user.name=vm commit -q -m "$1 $(date -Is)" || true
}

say "===== P1: re-tune at R_comm=${R_COMM} (K=${K_UAVS}, capacity=${CAPACITY}) ====="

# The feature cache is keyed on (N, data.seed, subsample). Build a fresh one
# rather than borrowing another sweep's: loading features built at a different
# subsample would silently train every trial on the wrong rows.
if [[ ! -f "$CACHE/N50/img_features.npy" ]]; then
    say "warmup — building feature caches for N=$NVALS at subsample=$SUB"
    python scripts/tune_weights.py --method fedcs --n-trials 1 \
        --r-comm $R_COMM --k-uavs $K_UAVS --capacity $CAPACITY \
        --n-values $NVALS --seeds $SEEDS --n-rounds 2 --subsample $SUB \
        --n-parallel-jobs-per-trial 1 --feature-cache-dir "$CACHE" \
        --study-name warmup_cache_5km \
        --storage "sqlite:///$(pwd)/results/hpo_5km/warmup.db" \
        || { say "WARMUP FAILED — every study below would score -inf; aborting"; exit 1; }
    commit_block "P1 warmup"
else
    say "warmup skipped — $CACHE already populated"
fi

run_study() {
    local method="$1" trials="$2"
    say "--- tuning $method ($trials trials) at R_comm=$R_COMM ---"
    python scripts/tune_weights.py --method "$method" --n-trials "$trials" \
        --r-comm $R_COMM --k-uavs $K_UAVS --capacity $CAPACITY \
        --n-values $NVALS --seeds $SEEDS --n-rounds $ROUNDS --subsample $SUB \
        --n-parallel-jobs-per-trial $PAR --feature-cache-dir "$CACHE" \
        --study-name "w5k_${method}" --storage "$STORAGE" \
        || say "$method FAILED"
    say "--- transfer check: $method ---"
    python scripts/tune_weights.py --method "$method" --transfer-check \
        --r-comm $R_COMM --k-uavs $K_UAVS --capacity $CAPACITY \
        --n-values $NVALS --seeds $SEEDS --n-rounds $ROUNDS --subsample $SUB \
        --n-parallel-jobs-per-trial $PAR --feature-cache-dir "$CACHE" \
        --study-name "w5k_${method}" --storage "$STORAGE" \
        || say "$method transfer-check FAILED"
    commit_block "P1 $method"
}

# proposed_hfl gets the full space (recipe + its own selector constants); the
# baselines get the recipe only, because they have no analogue of
# SEL_STATIC_BLEND or the utility weights. Same asymmetry as the 2026-08-05 run,
# and it favours the proposed method, which is the safe direction for a
# result that is currently negative.
run_study proposed_hfl 50
for m in fedcs oort flat_fl; do
    run_study "$m" 30
done

say "===== P1 complete ====="
say "NEXT: review results/hpo_5km/*_leaderboard.csv and *_transfer.csv, then"
say "adopt winners into configs/tuned_weights.yaml as a separate deliberate"
say "edit and RERUN the comparison. Adoption is NOT automatic."
commit_block "P1 complete"
