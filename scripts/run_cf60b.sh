#!/usr/bin/env bash
# R2 then R3+R4: the eight missing arms, then the main-grid report over both trees.
# No `||` fallback on the sweep: a failed sweep must stop before any report runs
# on an incomplete grid.
set -uo pipefail
cd "$HOME/client-selection-hfl"; source .venv/bin/activate
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]] && export HF_TOKEN=$(grep -o "hf_[A-Za-z0-9]*" "$HOME/.hf_env" | head -1)
exec >> results/paper_full_cf60b.log 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== R2: eight arms under cf60 recipes ====="
if ! python -m uavbench run_paper_sim --config configs/paper_full_cf60b.yaml; then
  say "  !! SWEEP FAILED — no report on an incomplete grid"; exit 1
fi
say "--- collapse gate ---"
python scripts/gate_collapse.py results/paper_full_cf60b || say "  !! degenerate cells"
say "--- R3+R4: main grid report ---"
python scripts/main_grid_report.py > results/main_grid_report.txt 2>&1 \
  || say "  !! report FAILED"
git add -A -- results configs >/dev/null 2>&1
git diff --cached --quiet 2>/dev/null || \
  git -c user.name="Abbas Cherri" -c user.email="abbasbcherri@gmail.com" commit -q -m "Add eight cf60 arms"
say "===== complete ====="
