#!/usr/bin/env bash
# Radius sweep under the decisive table's recipes — the second mechanism axis.
set -uo pipefail
cd "$HOME/client-selection-hfl"; source .venv/bin/activate
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]] && export HF_TOKEN=$(grep -o "hf_[A-Za-z0-9]*" "$HOME/.hf_env" | head -1)
exec >> results/paper_coverage_cf60.log 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

say "===== radius sweep under cf60 recipes ====="
python -m uavbench run_coverage_sim --config configs/paper_coverage_cf60.yaml \
    || python -m uavbench run_paper_sim --config configs/paper_coverage_cf60.yaml \
    || say "  !! run FAILED"
say "--- collapse gate ---"
python scripts/gate_collapse.py results/paper_coverage_cf60 || say "  !! degenerate cells"
git add -A -- results configs >/dev/null 2>&1
git diff --cached --quiet 2>/dev/null || \
  git -c user.email=vm@local -c user.name=vm commit -q -m "Add cf60 radius sweep"
say "===== complete ====="
