# Client Selection for Hierarchical Federated Learning with UAV Placement

Research code for post-earthquake building-damage classification via
**hierarchical federated learning (HFL)** over UAV-relayed IoT clients, with
metaheuristic **3-D UAV placement** (PSO/GA and published literature
baselines) and **reputation-aware client selection**. All experiments run on
one real fused multi-modal dataset for the **2024 Noto Peninsula earthquake**
— the pipeline is real-data only (no synthetic experiment data; offline unit
tests inject a deterministic fixture through a documented seam).

## Layout

| Path | Contents |
|---|---|
| `src/uavbench/` | The live experimental package: placement optimizers (`optimizers/`), problem model + path-loss coverage (`problem/`), FL harnesses, selection, reputation, sweeps (`fl/`), metrics, statistics (`analysis/`), reporting (seed manifests, timing). |
| `src/hflsim/` | Data pipeline (HF streaming, GSI imagery, partitioning) + a **legacy** standalone simulator kept only for the `hflsim` CLI and the `UAVAggregator` bridge. |
| `configs/` | Every experiment as a YAML config (one-line ablations; resolved copies + seed manifests persisted next to results). |
| `tests/` | 340 offline tests (`pytest`), including CI invariants that pin the fairness of the paired-comparison design. |
| `REPORTS/` | Implementation references, data availability + hardware/runtime disclosures. |
| `scripts/` | Reproduction entry point and GCP wrappers. |

## Data

The single real dataset streams from HuggingFace `AbbasABC/HFL-Dataset`
(pinned revision in `src/hflsim/data/loader.py`), fusing Noto-2024 building
damage labels, USGS ShakeMap parameters, and on-demand GSI aerial imagery.
`HF_TOKEN` is required on first run; metadata/partition/tile caches under
`./data/` (never tracked in git) make later runs offline-capable. See
`REPORTS/data_availability.md` for licensing and the stated single-event
limitation.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact pins used for the paper
pip install -e ".[dev]"
pytest -q                              # 340 offline tests, no token needed
```

## Running experiments

```bash
uavbench smoke                 # Tier-1 placement benchmark, minutes on CPU
uavbench run          --config configs/tier1_core.yaml   # Tier-1 grid (7 methods x 3 scenarios x 30 seeds)
uavbench smoke_tier2                                     # real-data FL smoke (reduced subsample)
uavbench run_paper_sim --config configs/paper_full.yaml  # full system sim (11 methods x 3 N x seeds)
uavbench run_selection_sim                               # selection-rule isolation benchmark
uavbench run_stress_sweep                                # dropout / SNR / black-chip robustness sweep
uavbench significance --config <results-dir> --metric accuracy   # paired Wilcoxon, Holm-corrected
```

Every run writes its resolved config, a `seed_manifest.csv` (exact seeds,
written before the run starts), per-round parquet tables, and confusion
matrices. `scripts/reproduce_paper.sh [--smoke]` chains the full grid end to
end; see `REPORTS/hardware_and_runtime.md` for machine specs and wall-clock
expectations.

## Method comparison design

All placement methods score through one shared fitness/assignment path
(`problem/fitness.py`), all selection methods run in the otherwise-identical
pipeline (`_METHOD_CFG` in `fl/federated.py`), and instance seeds are
method-independent — so per-seed results are paired samples, verified by
`tests/uavbench/test_invariants.py`.
