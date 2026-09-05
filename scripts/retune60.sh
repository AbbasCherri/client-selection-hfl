#!/usr/bin/env bash
# Symmetric 60-round re-tune under client_full — the remedy for the recipe
# confound recorded in PREREGISTRATION.md on 2026-09-05.
#
# Every method gets the SAME treatment: --space recipe, 25 trials, 60 rounds,
# tuning seeds 20-22, N in {30,50}, subsample 0.2. Equal budget, equal space,
# equal horizon, equal seeds. proposed_hfl is tuned recipe-only here rather than
# on its full space so that no method receives more search than another; its
# fitness weights stay fixed, which is conservative for our claim.
#
# flat_fl is NOT in this list — it is already running in the other stream on the
# identical COMMON settings.
#
# 6 workers per trial pairs with the flat_fl stream's 6 to saturate 12 vCPUs.
# Separate optuna storage so the two streams never contend on one SQLite file.
set -uo pipefail
cd "$HOME/client-selection-hfl"; source .venv/bin/activate
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]] && export HF_TOKEN=$(grep -o "hf_[A-Za-z0-9]*" "$HOME/.hf_env" | head -1)
exec >> results/retune60.log 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

S="sqlite:///$(pwd)/results/hpo_5km/weights_retune60.db"
C="results/hpo_cache_5km"
COMMON="--r-comm 5000 --k-uavs 20 --capacity 6 --fusion-owner client_full \
        --n-values 30 50 --seeds 20 21 22 --n-rounds 60 --subsample 0.2 \
        --space recipe --n-parallel-jobs-per-trial 6 \
        --feature-cache-dir $C --storage $S"

say "===== symmetric 60-round retune under client_full ====="

# Headline-carrying methods first: if anything dies, these exist.
for m in fedcs oort proposed_hfl hfl_no_selection rep_cap fair_mab power_of_choice; do
  say "TUNE $m — 25 trials, 60 rounds, recipe space"
  python scripts/tune_weights.py --method "$m" --n-trials 25 $COMMON \
      --study-name "cf60_$m" || say "  !! $m tuning FAILED"
  say "TRANSFER $m"
  python scripts/tune_weights.py --method "$m" --transfer-check $COMMON \
      --study-name "cf60_$m" || say "  !! $m transfer FAILED"
  git add -A -- results >/dev/null 2>&1
  git diff --cached --quiet 2>/dev/null || \
    git -c user.email=vm@local -c user.name=vm commit -q -m "Add $m 60round tuning"
done

say "===== retune complete ====="
