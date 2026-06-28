# uav-pso-bench — PSO 3D UAV-Placement Benchmark

A self-contained research benchmarking codebase that evaluates a **Particle Swarm Optimizer
(PSO)** for **3D UAV placement** in a hierarchical federated-learning (HFL) system for
post-earthquake building-damage classification.

It lives in its own subdirectory of the parent `client-selection-hfl` repo and does **not**
modify any existing files. Tier-2 (deferred) will reuse the parent repo's `data_loader.py`
real seismic dataset and `models.py` fusion head; Tier-1 is fully standalone.

## Two evaluation tiers

- **Tier 1 — Standalone placement benchmark (this pass).** Placement is treated as a pure
  optimization problem. PSO is compared against baselines on an *identical* fitness, *identical*
  greedy assignment, *identical* evaluation budget (P·G_max = 20,000 evals), and *identical*
  problem instances (paired comparison).
- **Tier 2 — In-the-loop FL evaluation (deferred).** Each placement method is plugged into a
  simplified federated loop to measure downstream model accuracy/convergence.

See [HARDWARE.md](HARDWARE.md) for the CPU/RAM/disk budget this is designed against.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes the `uavbench` package importable (src layout)
```

Without the editable install, prefix commands with `PYTHONPATH=src`
(e.g. `PYTHONPATH=src python -m uavbench smoke`).

## Run

```bash
# Fast end-to-end smoke run (minutes): PSO + GA + centroid + static, table + convergence plot
python -m uavbench smoke

# Full Tier-1 grid from a config
python -m uavbench run    --config configs/tier1_core.yaml
python -m uavbench analyze --config configs/tier1_core.yaml
python -m uavbench plot    --config configs/tier1_core.yaml

# Remove results/
python -m uavbench clean
```

If the package is installed (`pip install -e .`) the `uavbench` console script is equivalent
to `python -m uavbench`.

## What's implemented (Tier-1 core)

- Shared problem definition: `ProblemInstance`, value score `V_i(t)`, greedy value-sorted
  capacity-aware assignment, the scalarized fitness `F(X)`, and a rotary-wing energy model.
- Scenario generator: uniform, clustered (Gaussian-mixture), epicenter-biased.
- Our PSO (full spec, constriction + ring topology + per-dim clamp + stagnation reinit +
  turbulence + value-weighted k-means++ seeding) with every design choice as a config toggle.
- Baselines: GA (incumbent), centroid / value-weighted k-means, random, static.
- Parallel runner, Tier-1 metrics, convergence plotting.

## Deferred

DE / GWO / ABC / standard-PSO control / geometric-circle / exhaustive baselines; the statistics
module (Friedman, Wilcoxon+Holm, effect sizes); ablation and scalability configs; and all of
Tier-2.

## Layout

```
src/uavbench/
  problem/   instance.py value.py assignment.py fitness.py energy.py
  optimizers/ base.py pso.py ga.py heuristics.py
  metrics/   placement.py
  runner.py plotting.py cli.py
configs/  smoke.yaml tier1_core.yaml
tests/
```
