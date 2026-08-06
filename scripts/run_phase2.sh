#!/usr/bin/env bash
# Phase 2 — the paper's headline sweep, re-run at validation-selected recipes.
#
# Why this re-run exists (REPORTS/rigor_plan_2026-08.md §0.2 / §0.3b):
#   * every hyperparameter behind the previous numbers was chosen by maximising
#     macro_f1 on the reported TEST split;
#   * that one recipe was fit to proposed_hfl and imposed on all five
#     literature baselines, so the comparison was tuned vs untuned.
# Both are fixed: recipes now come from per-method studies scored on
# val_macro_f1 with tuning seeds disjoint from the evaluation seeds, and
# sweep._paper_job applies each method's own recipe.
#
# SELF-STOPS ON SUCCESS. The previous three runners deliberately did not, and
# the VM idled ~13 h, ~10 h and ~14 h waiting to be noticed — far more waste
# than any crash they were protecting against. A failure still leaves the
# instance up for inspection; only a clean finish shuts it down.
#
# Usage:  nohup ./scripts/run_phase2.sh &
set -uo pipefail          # not -e: a failing block must not skip the commit
cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG=results/phase2.log
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

say "===== Phase 2: paper_full at val-selected per-method recipes ====="

# Fail fast and loudly if the recipes are not actually reaching the jobs: a
# 40 h sweep that silently ran every method on the base recipe would look
# completely normal in its output and would have to be thrown away.
python tests/sanity_checks/check_sweep_resume.py || { say "RESUME/RECIPE CHECKS FAILED — aborting"; exit 1; }

python -m uavbench run_paper_sim --config configs/paper_full.yaml
STATUS=$?
commit_block "paper-full-v2"

if [[ $STATUS -ne 0 ]]; then
    say "===== Phase 2 FAILED (exit $STATUS) — VM left RUNNING for inspection ====="
    exit $STATUS
fi

say "===== Phase 2 complete — stopping the VM ====="
say "NEXT: pull results, then Phase 5 e2e (~12 h)."
sudo shutdown -h +2 "Phase 2 complete"
