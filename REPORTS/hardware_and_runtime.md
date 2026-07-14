# Hardware and Runtime Disclosure

## Machines

| Role | Machine | CPU | RAM | Notes |
|---|---|---|---|---|
| Development / smoke runs | Dell Latitude 5540 | Intel i7-1355U (10c/12t) | 32 GB | CPU-only; all harnesses runnable |
| Full experimental grids | GCP `n1-standard-12` | 12 vCPU | 45 GB | via `scripts/run_gcp.sh` / `scripts/run_paper_sim.sh` / `scripts/run_selection_gcp.sh` (self-terminating) |

No GPU is used anywhere: the CPU-feasibility claim is backed by the
measured wall-clock numbers below, not an architectural argument.

## Parallelism per harness

Each sweep worker pins `torch.set_num_threads(1)` so total active threads =
`n_workers` × 1 (no BLAS thrash). `n_workers` per checked-in config:

| Config | Harness | n_workers |
|---|---|---|
| `configs/tier1_core.yaml` | Tier-1 placement grid | 8 |
| `configs/paper_full.yaml` | full paper sim (N × method × seed) | 12 |
| `configs/selection_isolation.yaml` | selection isolation | (see config) |
| `configs/tier2_sweep.yaml` | N-scalability sweep | (see config) |
| `configs/stress_test.yaml` | stress-test sweep | 8 |

`UAVBENCH_N_WORKERS` overrides the Tier-1 worker count per machine.

## Where the timing numbers come from

Wall-clock is instrumented at two levels and persisted with every run:

- **Per optimizer run** — `wall_time_s` column in Tier-1 `runs.parquet`
  (timed around `Optimizer.optimize`).
- **Per FL round** — `round_time_s` column in every rounds table
  (`tier2_rounds.parquet`, `fullsim_rounds.parquet`,
  `selection_rounds.parquet`, `stress_rounds.parquet`).

The per-method aggregate (mean/std/total seconds) is printed by
`uavbench analyze` / `run_tier2` / `run_paper_sim` / `run_stress_sweep`
via `uavbench.reporting.summarize_wall_clock`, and should be quoted in the
paper's runtime disclosure.

## Fill in after the final grid run

Record the totals from `results/reproduce_paper.log` and the wall-clock
summaries here before submission:

| Harness | Grid size | Total wall-clock |
|---|---|---|
| Tier-1 core (`tier1_core.yaml`) | 7 methods × 3 scenarios × 30 seeds | _TBD_ |
| Paper full sim (`paper_full.yaml`) | 11 methods × 3 N × 3 seeds | _TBD (previous 9-method run: ~4–7 h on n1-standard-12)_ |
| Selection isolation | modes × N × seeds | _TBD_ |
| N-scalability sweep | methods × 6 N | _TBD_ |
| Stress-test sweep | 11 cells × 4 methods × 5 seeds | _TBD_ |

## Environment

- Python 3.13.14, exact package pins in `requirements-lock.txt`
  (`requirements.txt` / `pyproject.toml` remain floor-pinned for
  installability).
- Reproduction entry point: `scripts/reproduce_paper.sh` (add `--smoke`
  for a minutes-scale end-to-end check).
