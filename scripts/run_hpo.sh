#!/usr/bin/env bash
# 0.3b — per-method hyperparameter search on VALIDATION data.
#
# Fixes the tuning asymmetry: the training recipe (lr, schedule, tau, momentum)
# was fit to proposed_hfl with 120 Optuna trials and then imposed on every
# baseline, so the headline comparison was tuned-vs-untuned. Each method now
# gets its own study, scored on val_macro_f1 with tuning seeds (20-22) disjoint
# from the evaluation seeds (0-19).
#
# BLOCKS PHASE 2. Re-running paper_full before this lands would just reproduce
# the test-set-selected hyperparameters at ~50 h of cost.
#
# WARMUP IS NOT OPTIONAL. Python 3.14 defaults to the 'forkserver' start method,
# so a DataLoader with num_workers>0 cannot spawn from inside a joblib worker —
# every trial dies with AttributeError("'Process' object has no attribute
# 'env'") and scores -inf. The warmup runs one trial at n_jobs=1 (joblib runs
# in-process at n_jobs=1), which builds the ResNet feature cache for each N in
# the MAIN process. Every later trial just loads the .npy.
#
# Usage:  sudo nohup ./scripts/run_hpo.sh &
set -uo pipefail          # not -e: one method failing must not skip the rest
cd "$(dirname "$0")/.."

# results/ is root-owned so this runs under sudo, which does not inherit the venv.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG=results/hpo.log
mkdir -p results/hpo
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -Is)] $*"; }

CACHE=results/hpo_cache
SEEDS="20 21 22"
NVALS="30 50"
SUB=0.2
ROUNDS=20
PAR=6

commit_block() {
    git add -A -- results || true
    if git diff --cached --quiet; then
        say "  nothing new to commit"
    else
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add $1 results $(date -Is)" || true
        say "  committed $(git rev-parse --short HEAD)"
    fi
}

say "===== 0.3b per-method HPO ====="

# Fresh cache dir rather than reusing another sweep's: the feature cache is keyed
# on (N, data.seed, subsample), and silently loading features built at a
# different subsample would train every trial on the wrong rows.
if [[ ! -f "$CACHE/N50/img_features.npy" ]]; then
    say "warmup — building feature caches for N=$NVALS at subsample=$SUB (in-process)"
    python scripts/tune_weights.py --method fedcs --n-trials 1 \
        --n-values $NVALS --seeds $SEEDS --n-rounds 2 --subsample $SUB \
        --n-parallel-jobs-per-trial 1 \
        --feature-cache-dir "$CACHE" --study-name warmup_cache \
        --storage sqlite:///results/hpo/warmup.db \
        || say "WARMUP FAILED — the studies below will all score -inf"
    commit_block "hpo-warmup"
else
    say "warmup skipped — $CACHE already populated"
fi

# proposed_hfl searches the full space (recipe + our selector's own constants);
# the baselines search the recipe only, because they have no analogue of
# SEL_STATIC_BLEND / the utility weights and tuning those for them would be
# meaningless rather than generous.
run_study() {
    local method="$1" trials="$2"
    say "--- tuning $method ($trials trials) ---"
    python scripts/tune_weights.py --method "$method" --n-trials "$trials" \
        --n-values $NVALS --seeds $SEEDS --n-rounds $ROUNDS --subsample $SUB \
        --n-parallel-jobs-per-trial $PAR --feature-cache-dir "$CACHE" \
        || say "$method FAILED"
    say "--- transfer check: $method ---"
    python scripts/tune_weights.py --method "$method" --transfer-check \
        --n-values $NVALS --seeds $SEEDS --n-rounds $ROUNDS --subsample $SUB \
        --n-parallel-jobs-per-trial $PAR --feature-cache-dir "$CACHE" \
        || say "$method transfer-check FAILED"
    commit_block "hpo-$method"
}

run_study proposed_hfl 50
for m in fedcs rep_cap fair_mab oort power_of_choice; do
    run_study "$m" 30
done

say "===== HPO complete — VM left RUNNING ====="
say "NEXT: review results/hpo/*_leaderboard.csv + *_transfer.csv, adopt winners"
say "into configs/tuned_weights.yaml, THEN launch Phase 2 paper_full."
