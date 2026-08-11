# Paper data manifest — what exists, what is missing, what was decided against

Audited 2026-08-11. One row per table or figure the submission needs, with the
file it comes from and its status. **A row marked GAP means the paper cannot be
submitted with that element until it is filled or explicitly dropped.**

Status vocabulary: `have` (fresh, corrected regime) · `running` · `GAP` ·
`dropped` (deliberate, with reason) · `void` (exists but unusable).

Every `have` row traces to a row in `results_provenance.md`; that file remains
the authority on what each number means.

---

## 1. Setup and protocol

| # | Element | Source | Status |
|---|---|---|---|
| 1.1 | Dataset characterisation — class distribution, per-client skew, geographic spread | `scripts/dataset_stats.py` → `results/dataset_stats/*.csv` | have (2026-08-11) |
| 1.2 | Coherent altitude–radius curve (the Defect-1 figure) | `scripts/plot_coherent_band.py` → `results/protocol_figs/coherent_band.{png,pdf}` | have (2026-08-11) |
| 1.3 | Degeneracy-gate definition + calibration set (3 runs at 0.287/0.266/0.262 vs a 0.2245 floor) | `src/uavbench/analysis/collapse.py`, `tests/sanity_checks/check_collapse_guard.py`, provenance row 2026-08-08 | have |
| 1.4 | Constant-predictor floor, closed form | `collapse.constant_predictor_macro_f1` | have |
| 1.5 | Seeds, resolved configs, hardware | `seed_manifest.csv` + `config.*.resolved.yaml` in every results dir | have |
| 1.6 | Reproduction script covering every experiment in the paper | `scripts/reproduce_paper.sh` | have (rewritten 2026-08-11; dependency-ordered, documents what it deliberately skips) |

## 2. Placement (Tier-1)

| # | Element | Source | Status |
|---|---|---|---|
| 2.1 | Tier-1 headline: 7 methods × 4 scenarios × 30 seeds, Holm + effect size + CI | `results/tier1_core/{runs,summary}.parquet`, `significance_final_fitness.csv` | have (2026-08-10) |
| 2.2 | Convergence curves | `results/tier1_core/convergence_*.png` (8) | have |
| 2.3 | Saturating-capacity, heterogeneous-fleet, warm-start ablations | `results/tier1_{equal_radius,regime_hetero,warmprev}/` | have |
| 2.4 | MCLP near-optimality reference | `results/tier1_core/mclp_reference.csv` | have (2026-08-11) — grid-convergence 20/30/45, all 12 proven optimal |
| 2.5 | Altitude behaviour per method (the "vertical decision is live" evidence) | `mean_altitude_m` in `runs.parquet`, `scripts/gate_altitude.py` output | have |

## 3. Federated learning

| # | Element | Source | Status |
|---|---|---|---|
| 3.1 | FL main table: 13 methods × N ∈ {30,50,100,200} × 10 seeds | `results/paper_full/paper_sweep_rounds.parquet`, `significance_macro_f1.csv` | have |
| 3.2 | Accuracy / macro-F1 vs round, per N | `results/paper_full/paper_*_vs_rounds_N*.png` (11 figures) | have |
| 3.3 | Per-class F1 and confusion | `per_class_f1.csv`, `confusion.parquet` | have |
| 3.4 | Operational metrics — movement energy, comm cost, coverage | `operational_summary.csv` | have |
| 3.5 | Convergence: rounds-to-target | `rounds_to_target` column | have |
| 3.6 | Fleet-size sweep, 17 methods × 5 K | `results/paper_uav_count/uav_sweep_rounds.parquet` | have (data) |
| 3.7 | Fleet-sweep figures | `results/paper_uav_count/uav_sweep_*.png` (4) | have — `plot_uav_sweep` added 2026-08-11; the harness had no plotting path at all |
| 3.8 | Coverage-constrained sweep, 17 methods × 6 radii | `results/paper_coverage_v5/` + 3 figures | have |

## 4. Ablations, interventions, controls

| # | Element | Source | Status |
|---|---|---|---|
| 4.1 | Class-realism ablation (6 arms) | `results/class_realism_*` | have — **report with its MDE; the selection contrast is underpowered** |
| 4.2 | v6 2×2 (C1, C2, C1+C2) + verdict | `results/v6_*`, `results/v6_verdict.txt` | have |
| 4.3 | v6 refactor no-op control (diff = 0.0) | `results/v6_control/` | have |
| 4.4 | C3 screen (H-B rejected, H-A survives) | `results/c3_screen*`, `c3_screen_verdict.txt` | have |
| 4.5 | C3 confirmatory arm + verdict | `results/v6_c3_disjoint/`, `c3_verdict.txt` | have |
| 4.6 | Coverage-causality analysis (observational slope vs interventional delivery) | `scripts/coverage_causality.py` → `results/coverage_causality/{mediators,causality}.csv` | have (2026-08-11) |
| 4.7 | Placement-geometry columns (multiplicity, unique-cover, separation) | recorded per round since `46f601d3` | have (v6/C3 arms only; v5 baselines predate the columns) |
| 4.8 | Capacity-floor diagnostic | `results/probe_topology/` | have |
| 4.9 | Roster-construction control | `results/roster_control/` | running |
| 4.10 | C2 at n=25 + client-count generalisation | `results/c2_power_*` | queued |
| 4.11 | Power analysis / MDE for every null | `scripts/power_analysis.py` | have |
| 4.12 | Frozen-vs-end-to-end validation | `results/e2e_centralized/e2e_comparison.csv` | have — survives the void, see provenance carve-out |

## 5. Deliberately dropped

| Element | Reason |
|---|---|
| Selection-isolation sweep (`results/selection_isolation`) | Void (2026-08-06 regime). **Not being regenerated:** in `paper_full` every selector already runs on identical placement, so selection is isolated there, and the corrected selection result (proposed loses to FedCS/Oort) comes from that table. Regenerating a 45 h experiment to re-answer a question `paper_full` answers would not change any claim. State in the paper that selection is isolated within `paper_full` by construction. |
| Stress / robustness grid (`results/stress_test`, `stress_selection`) | Void. Dropped for cost. The fleet sweep (5 K), coverage sweep (6 radii) and N-sweep (4 client counts) already vary operating conditions across 27 cells, which carries the robustness argument. **If a reviewer demands perturbation robustness this is the first thing to re-run** — flag as a limitation rather than pretending it exists. |

## 6. What must happen before submission

Ordered by whether the paper can go without it.

**All blocking items are closed as of 2026-08-11.** 4.6, 1.6, 1.1, 1.2, 2.4 and
3.7 were filled without any new simulation compute — they were scripts and
figures over data already in hand.

Remaining, and neither blocks a draft:

1. **4.9 roster-construction control** and **4.10 C2 at n=25 + client-count
   generalisation** are running; both are pre-registered and land on their own.
2. **Writing.** Every table and figure the paper needs now exists or is in
   flight. What is left is prose, against `REPORTS/paper_reframe_2026-08-10.md`.

Two caveats that must survive into the text, because they are easy to lose:
* every null carries its MDE (`scripts/power_analysis.py`) — n=10 could not
  detect effects below ~0.034, so most per-cell nulls are "not detected", not
  "absent";
* the coverage-causality claim rests on the **aggregate across K**, never on a
  single cell.
