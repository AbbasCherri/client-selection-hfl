#!/usr/bin/env bash
# The decisive table under symmetric 60-round recipes.
set -uo pipefail
cd "$HOME/client-selection-hfl"; source .venv/bin/activate
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]] && export HF_TOKEN=$(grep -o "hf_[A-Za-z0-9]*" "$HOME/.hf_env" | head -1)
exec >> results/paper_full_cf60.log 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== decisive table: paper_full_cf60 ====="
python -m uavbench run_paper_sim --config configs/paper_full_cf60.yaml \
    || say "  !! run FAILED"
say "--- collapse gate ---"
python scripts/gate_collapse.py results/paper_full_cf60 \
    || say "  !! degenerate cells"
git add -A -- results configs >/dev/null 2>&1
git diff --cached --quiet 2>/dev/null || \
  git -c user.email=vm@local -c user.name=vm commit -q -m "Add cf60 decisive table"
say "===== complete ====="
