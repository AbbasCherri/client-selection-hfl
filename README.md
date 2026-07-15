# Client Selection for Hierarchical Federated Learning with UAV Placement

Research code for post-earthquake building-damage classification via
hierarchical federated learning over UAV-relayed IoT clients, with
metaheuristic 3-D UAV placement and reputation-aware client selection.
Design rationale and implementation details:
`REPORTS/master_implementation_reference.md`. Which config produced which
result: `REPORTS/results_provenance.md`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact pins used for the paper
pip install -e .
```

Python 3.13. Sanity checks are plain scripts, run manually before trusting
a batch of results (offline, no HF token needed):

```bash
python tests/sanity_checks/run_all.py
```

## Data

Experiments stream the real fused multi-modal dataset for the 2024 Noto
Peninsula earthquake from HuggingFace `AbbasABC/HFL-Dataset`, pinned to
revision `6cf97c900445e080e61cb45e1aa72515d3ff1de8` (default in
`src/hflsim/data/loader.py`, overridable via `HF_DATASET_REVISION`).
Export `HF_TOKEN` before the first run; metadata/partition/tile caches
under `./data/` (gitignored, never redistributed) make later runs
offline-capable. GSI aerial tiles are fetched on demand and cached under
`./data/tile_cache` (`HFL_TILE_CACHE`).

## Running experiments

`uavbench` subcommands (each writes into the `results_dir` named by its
config):

```bash
uavbench smoke                                             # fast Tier-1 end-to-end check
uavbench run              --config configs/tier1_core.yaml # Tier-1 placement grid
uavbench analyze          --config configs/tier1_core.yaml # summary table for saved Tier-1 runs
uavbench plot             --config configs/tier1_core.yaml # Tier-1 convergence figures
uavbench run_tier2        --config configs/tier2_fl.yaml   # Tier-2 FL benchmark
uavbench smoke_tier2                                       # Tier-2 smoke, real data at reduced subsample
uavbench run_paper_sim    --config configs/paper_full.yaml # full paper system sim (N x method x seed)
uavbench run_selection_sim                                 # selection-rule isolation benchmark
uavbench run_sweep        --config configs/tier2_sweep.yaml# N-scalability sweep
uavbench run_stress_sweep --config configs/stress_test.yaml# dropout / SNR / black-chip robustness sweep
uavbench significance     --config <results-dir-or-config> --metric accuracy  # paired Wilcoxon, Holm-corrected
uavbench clean            [--config <config>]              # remove results
```

`scripts/reproduce_paper.sh [--smoke]` chains the full grid end to end,
logging to `results/reproduce_paper.log`; every step checkpoints per job
and resumes on re-run instead of restarting from scratch if interrupted.
`scripts/run_gcp.sh` wraps it for a self-terminating GCP VM run (see
`scripts/gcp_setup.sh` for environment setup). A legacy `hflsim` CLI also
exists for the pre-`uavbench` standalone simulator.

## Configs

Every experiment is a YAML file under `configs/`:

| Config | Harness |
|---|---|
| `smoke.yaml` | reduced Tier-1 grid (used by `uavbench smoke` path in the reproduce script) |
| `tier1_core.yaml` | Tier-1 placement grid (7 methods x 3 scenarios x 30 seeds) |
| `tier2_fl.yaml` / `tier2_reduced.yaml` | Tier-2 FL benchmark (full / smoke) |
| `tier2_sweep.yaml` | N-scalability sweep |
| `paper_full.yaml` | full system sim (11 methods x 3 N x seeds) |
| `selection_isolation.yaml` | selection-rule isolation |
| `stress_test.yaml` | robustness stress sweep |

## Outputs

Runs write into `results/<name>/`: a resolved config YAML, a
`seed_manifest.csv` (exact seeds, written before the run starts),
per-round/per-run parquet tables, `confusion.parquet`, and figures.
Every number destined for the paper must have a row in
`REPORTS/results_provenance.md`.
