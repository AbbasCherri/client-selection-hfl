#!/usr/bin/env bash
# One-command reproduction of every figure and table in the paper.
#
# Usage:
#   scripts/reproduce_paper.sh           # full grid — days of compute
#   scripts/reproduce_paper.sh --smoke   # reduced end-to-end check (real data, small)
#
# Requirements: the pinned environment (pip install -r requirements.txt, then
# pip install -e .) and HF_TOKEN exported for the real-data harnesses. Every
# step logs to results/reproduce_paper.log and each harness writes its own
# seed_manifest.csv + config.*.resolved.yaml next to its outputs.
#
# Interrupted runs: every simulation step checkpoints per job and resumes, so
# re-running this script unchanged continues from where it stopped.
#
# ORDER MATTERS — it is a dependency order, not a narrative one:
#   * the fleet sweep (step 3) must precede v6, C3 and the C2 power run, which
#     reuse its `moon2022` / `mclp_place` / `flat_fl` cells as paired baselines
#     instead of recomputing them
#   * the v6 C1 arm must precede the coverage-causality analysis, which needs
#     both the observational slope and the intervention
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT RUN, and why (see
# REPORTS/paper_data_manifest.md §5):
#   * selection-isolation sweep — `paper_full` already isolates selection, since
#     every selector there runs on identical placement. Regenerating a 45 h
#     experiment to re-answer that would change no claim.
#   * stress / robustness grid — dropped for cost. The fleet, coverage and N
#     sweeps vary conditions across 27 cells. Stated as a limitation in the
#     paper rather than silently omitted.
#
# Everything reported in the paper must trace to a row in
# REPORTS/results_provenance.md.

set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

mkdir -p results
LOG=results/reproduce_paper.log
exec > >(tee -a "$LOG") 2>&1

echo "=== reproduce_paper.sh started $(date -Is) (smoke=$SMOKE) ==="
step() { echo; echo "--- [$(date -Is)] $* ---"; }

# ---------------------------------------------------------------- 0. protocol
# Analytic and dataset-level artifacts. Cheap, and they gate everything else:
# if the altitude band and R_comm are incoherent, no simulation below is
# interpretable (Defect 1).
step "Protocol figure — coherent altitude-radius band"
python scripts/plot_coherent_band.py --out results/protocol_figs

step "Dataset characterisation (class skew, per-client skew, geography)"
python scripts/dataset_stats.py --n 200 --out results/dataset_stats

step "Sanity checks — the guards every later claim rests on"
for chk in check_altitude_band check_tier1_link check_collapse_guard \
           check_coverage_mode check_disjoint_coverage check_uav_weight_mode \
           check_shard_diversity check_roster_control check_roster_width \
           check_seed_alias check_tier1_resume check_placement_geometry; do
    [[ -f "tests/sanity_checks/${chk}.py" ]] && python "tests/sanity_checks/${chk}.py"
done

if [[ $SMOKE -eq 1 ]]; then
    step "Tier-1 smoke"
    python -m uavbench run --config configs/smoke.yaml
    python -m uavbench analyze --config configs/smoke.yaml
    step "Coverage-sweep smoke"
    python -m uavbench run_coverage_sweep --config configs/coverage_smoke.yaml
    step "End-to-end centralized smoke"
    python scripts/e2e_centralized.py --subsample 0.03 --n 20 --epochs 1 \
        --out results/e2e_smoke
    echo; echo "=== smoke finished $(date -Is) ==="
    exit 0
fi

# ------------------------------------------------------------- 1. placement
# run_tier1_v5.sh rather than bare `uavbench run`: it deletes derived artifacts
# before recomputing (a significance CSV is not overwritten by a rerun and will
# otherwise sit stale beside fresh primary data), gates the vertical search, and
# runs plot + significance itself.
step "Tier-1 placement benchmark + ablations (scripts/run_tier1_v5.sh)"
./scripts/run_tier1_v5.sh

step "Tier-1 MCLP optimality reference (grid-convergence 20/30/45)"
python -m uavbench mclp --config configs/tier1_core.yaml --n-seeds 3

# ------------------------------------------------------------------ 2. FL main
step "FL main results (configs/paper_full.yaml)"
python -m uavbench run_paper_sim --config configs/paper_full.yaml
python scripts/gate_collapse.py results/paper_full || echo "  (degenerate cells — see gate output)"
step "FL significance (macro_f1 + accuracy, Holm within N)"
python -m uavbench significance --config results/paper_full --metric macro_f1 --reference proposed_hfl --correction-scope group
python -m uavbench significance --config results/paper_full --metric accuracy --reference proposed_hfl --correction-scope group

# --------------------------------------------------------------- 3. sweeps
# The fleet sweep is a PREREQUISITE for steps 5-7, which reuse its cells.
step "Fleet-size sweep (configs/paper_uav_count.yaml) — baseline for v6/C3/C2"
python -m uavbench run_uav_sweep --config configs/paper_uav_count.yaml
python scripts/gate_collapse.py results/paper_uav_count || echo "  (degenerate cells — expected at small K)"
python -m uavbench plot --config configs/paper_uav_count.yaml || echo "  (plot failed, non-fatal)"

step "Coverage-constrained sweep (configs/paper_coverage.yaml)"
python -m uavbench run_coverage_sweep --config configs/paper_coverage.yaml
python scripts/gate_collapse.py results/paper_coverage_v5 || echo "  (degenerate cells — expected below 3 km)"

# ------------------------------------------------------------ 4. ablations
step "Class-realism ablation (6 arms)"
./scripts/run_class_realism.sh

step "Roster-construction control (isolating ablation for the N-sweep)"
./scripts/run_roster_control.sh

# ------------------------------------------------- 5-7. interventions
step "v6 2x2 — C1 reachable x C2 diversity, control first"
./scripts/run_v6.sh

step "C3 screen — which geometric hypothesis survives"
./scripts/run_c3_screen.sh

step "C3 confirmatory arm — redundancy-discounted coverage"
./scripts/run_c3.sh

step "C2 at n=25 + client-count generalisation"
./scripts/run_c2_power.sh

# ---------------------------------------------------------- 8. validation
step "End-to-end centralized (frozen vs trainable ResNet-18)"
python scripts/e2e_centralized.py --subsample 0.2 --n 200 --epochs 15 \
    --out results/e2e_centralized

# ------------------------------------------------------------ 9. analyses
# These produce paper numbers and must not live in a scratch directory.
step "Coverage causality — observational slope vs C1's intervention"
python scripts/coverage_causality.py --out results/coverage_causality

step "Power analysis — minimum detectable effect for every reported null"
python scripts/power_analysis.py | tee results/power_analysis.txt

# ----------------------------------------------------------- 10. artifact
step "Staging results/paper_artifact"
ARTIFACT=results/paper_artifact
rm -rf "$ARTIFACT"          # else a stale copy of a deleted table survives here
mkdir -p "$ARTIFACT"
find results -maxdepth 2 \
    \( -name "seed_manifest.csv" -o -name "config.*.resolved.yaml" \
       -o -name "significance*.csv" -o -name "*_verdict.txt" \) \
    -not -path "$ARTIFACT/*" | while read -r f; do
    cp "$f" "$ARTIFACT/$(dirname "${f#results/}" | tr '/' '_')_$(basename "$f")"
done
echo "Artifact staged at $ARTIFACT ($(ls "$ARTIFACT" | wc -l) files)"

echo
echo "=== reproduce_paper.sh finished $(date -Is) ==="
