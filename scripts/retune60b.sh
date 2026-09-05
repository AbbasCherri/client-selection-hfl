#!/usr/bin/env bash
# Second retune stream — takes the TAIL of scripts/retune60.sh's method list so
# the two streams never touch the same optuna study.
#
#   stream A (retune60.sh) : fedcs oort proposed_hfl hfl_no_selection rep_cap ...
#   stream B (this file)   : ... fair_mab power_of_choice
#
# Stream A is killed once it finishes rep_cap, before it can reach fair_mab.
# Each method therefore receives exactly 25 trials from exactly one stream, which
# is the whole point — an asymmetric search budget would reintroduce the very
# confound this retune exists to remove.
#
# Identical COMMON settings to stream A. Own storage file, own log.
set -uo pipefail
cd "$HOME/client-selection-hfl"; source .venv/bin/activate
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]] && export HF_TOKEN=$(grep -o "hf_[A-Za-z0-9]*" "$HOME/.hf_env" | head -1)
exec >> results/retune60b.log 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

S="sqlite:///$(pwd)/results/hpo_5km/weights_retune60b.db"
C="results/hpo_cache_5km"
COMMON="--r-comm 5000 --k-uavs 20 --capacity 6 --fusion-owner client_full \
        --n-values 30 50 --seeds 20 21 22 --n-rounds 60 --subsample 0.2 \
        --space recipe --n-parallel-jobs-per-trial 6 \
        --feature-cache-dir $C --storage $S"

say "===== retune stream B: fair_mab, power_of_choice ====="

for m in fair_mab power_of_choice; do
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

say "===== stream B complete ====="
