#!/usr/bin/env bash
# One-command reproduction of every figure and table in the paper, from raw
# runs to significance tests, using the checked-in configs and seed manifests.
#
# Usage:
#   scripts/reproduce_paper.sh           # full grid (hours; see REPORTS/master_implementation_reference.md §18)
#   scripts/reproduce_paper.sh --smoke   # reduced end-to-end check (real data at small subsample)
#
# Requirements: the pinned environment (pip install -r requirements.txt,
# then pip install -e .) and, for the real-data harnesses, HF_TOKEN exported.
# Every step logs to results/reproduce_paper.log and each harness writes its
# own seed_manifest.csv + config.*.resolved.yaml next to its outputs.
#
# Interrupted runs: every step below (run/run_paper_sim/run_selection_sim/
# run_sweep/run_stress_sweep) checkpoints each (N, method/mode, seed) job as
# it finishes and skips + reloads already-finished jobs on the next attempt.
# If this script is killed (Ctrl-C, SSH drop, VM preemption), just re-run it
# unchanged — it resumes from the last completed job in the step it was on,
# then continues to the remaining steps, instead of starting over.

set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

mkdir -p results
LOG=results/reproduce_paper.log
exec > >(tee -a "$LOG") 2>&1

echo "=== reproduce_paper.sh started $(date -Is) (smoke=$SMOKE) ==="

if [[ $SMOKE -eq 1 ]]; then
    TIER1_CFG=configs/smoke.yaml
    STRESS_CFG=${STRESS_SMOKE_CFG:-configs/stress_test.yaml}
else
    TIER1_CFG=configs/tier1_core.yaml
    STRESS_CFG=configs/stress_test.yaml
fi

step() { echo; echo "--- [$(date -Is)] $* ---"; }

# 1. Tier-1 placement benchmark (PSO/GA/heuristics/literature baselines)
step "Tier-1 placement grid ($TIER1_CFG)"
python -m uavbench run --config "$TIER1_CFG"
python -m uavbench analyze --config "$TIER1_CFG"
python -m uavbench plot --config "$TIER1_CFG"
step "Tier-1 significance (final_fitness, paired Wilcoxon + effect size + CIs)"
python -m uavbench significance \
    --config "$(python -c "import yaml;print(yaml.safe_load(open('$TIER1_CFG'))['results_dir'])")" \
    --metric final_fitness
step "Tier-1 MCLP optimality reference (PSO % of MILP-optimal coverage)"
python -m uavbench mclp --config "$TIER1_CFG" --n-seeds 3

if [[ $SMOKE -eq 1 ]]; then
    # Reduced real-data stand-ins for the full harnesses.
    step "Tier-2 smoke (real, reduced)"
    python -m uavbench smoke_tier2
    step "Coverage-sweep smoke (2 R_comm × 2 methods × 1 seed)"
    python -m uavbench run_coverage_sweep --config configs/coverage_smoke.yaml
    step "End-to-end centralized smoke (tiny)"
    python scripts/e2e_centralized.py --subsample 0.03 --n 20 --epochs 1 \
        --out results/e2e_smoke
else
    # 2. Full paper system simulation (real data; needs HF_TOKEN)
    step "Full paper simulation (configs/paper_full.yaml)"
    python -m uavbench run_paper_sim --config configs/paper_full.yaml
    step "Paper-sim significance (accuracy + macro_f1, paired Wilcoxon)"
    python -m uavbench significance --config results/paper_full --metric accuracy --reference proposed_hfl --correction-scope group
    python -m uavbench significance --config results/paper_full --metric macro_f1 --reference proposed_hfl --correction-scope group

    # 3. Selection-isolation benchmark (real data)
    step "Selection isolation (configs/selection_isolation.yaml)"
    python -m uavbench run_selection_sim --config configs/selection_isolation.yaml
    step "Selection-isolation significance (accuracy + macro_f1)"
    python -m uavbench significance --config results/selection_isolation --metric accuracy --reference ucb --correction-scope group
    python -m uavbench significance --config results/selection_isolation --metric macro_f1 --reference ucb --correction-scope group

    # 4. N-scalability sweep (real data)
    step "N-scalability sweep (configs/tier2_sweep.yaml)"
    python -m uavbench run_sweep --config configs/tier2_sweep.yaml

    # 4b. Coverage-constrained sweep — placement quality vs macro_f1 as R_comm binds
    step "Coverage-constrained sweep (configs/paper_coverage.yaml)"
    python -m uavbench run_coverage_sweep --config configs/paper_coverage.yaml
    step "Coverage-sweep significance (macro_f1, per R_comm)"
    python -m uavbench significance --config results/paper_coverage --metric macro_f1 --reference proposed_hfl --correction-scope group

    # 4c. End-to-end centralized validation of the frozen-feature simplification
    step "End-to-end centralized (frozen vs trainable ResNet-18)"
    python scripts/e2e_centralized.py --subsample 0.2 --n 200 --epochs 15 \
        --out results/e2e_centralized
fi

# 5. Synthetic stress-test sweep (robustness evidence; no HF_TOKEN needed)
step "Stress-test sweep ($STRESS_CFG)"
python -m uavbench run_stress_sweep --config "$STRESS_CFG"
step "Stress-sweep significance (accuracy + macro_f1)"
STRESS_RESULTS="$(python -c "import yaml;print(yaml.safe_load(open('$STRESS_CFG'))['results_dir'])")"
python -m uavbench significance --config "$STRESS_RESULTS" --metric accuracy --reference proposed_hfl --correction-scope group
# macro_f1 is the primary reported metric (fl.target_metric) — it was missing
# here through the 2026-07-22 run, leaving the robustness claims on accuracy
# alone, which barely separates methods under an ~0.8 majority class.
python -m uavbench significance --config "$STRESS_RESULTS" --metric macro_f1 --reference proposed_hfl --correction-scope group

# 5b. Selection-pressure stress sweep — same grid with K*capacity < N so the
# client-selection rule binds (stress_test.yaml has slots == N, leaving
# selection inert). Skipped in smoke runs. See configs/stress_selection.yaml.
if [[ $SMOKE -ne 1 ]]; then
    step "Selection-pressure stress sweep (configs/stress_selection.yaml)"
    python -m uavbench run_stress_sweep --config configs/stress_selection.yaml
    step "Selection-pressure stress significance (accuracy + macro_f1)"
    SELSTRESS_RESULTS="$(python -c "import yaml;print(yaml.safe_load(open('configs/stress_selection.yaml'))['results_dir'])")"
    python -m uavbench significance --config "$SELSTRESS_RESULTS" --metric accuracy --reference proposed_hfl --correction-scope group
    python -m uavbench significance --config "$SELSTRESS_RESULTS" --metric macro_f1 --reference proposed_hfl --correction-scope group
fi

# 6. Stage the paper artifact: seed manifests + resolved configs + significance tables
step "Staging results/paper_artifact"
ARTIFACT=results/paper_artifact
mkdir -p "$ARTIFACT"
find results -maxdepth 2 \
    \( -name "seed_manifest.csv" -o -name "config.*.resolved.yaml" -o -name "significance.csv" -o -name "significance_*.csv" \) \
    -not -path "$ARTIFACT/*" | while read -r f; do
    dest="$ARTIFACT/$(dirname "${f#results/}" | tr '/' '_')_$(basename "$f")"
    cp "$f" "$dest"
done
echo "Artifact staged at $ARTIFACT ($(ls "$ARTIFACT" | wc -l) files)"

echo
echo "=== reproduce_paper.sh finished $(date -Is) ==="
