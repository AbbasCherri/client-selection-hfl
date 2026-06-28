# Hardware Target and Resource Budget

This benchmark is designed to run on a **Google Cloud `n1-standard-8`**:

| Resource | Budget | Design consequence |
|---|---|---|
| CPU | 8 vCPUs | **Run-level** parallelism with `joblib` across `n_workers` (default 8). We parallelize *independent* optimizer runs (method × scenario × seed), **not** inner optimizer iterations — coarse-grained scaling avoids per-iteration IPC overhead. |
| GPU | none | Everything runs on CPU. Tier-1 is pure NumPy/SciPy. Tier-2 (deferred) will use a PyTorch **CPU** build only. |
| RAM | ~30 GB | Tier-1 data structures are tiny (a swarm is a `P×3K` float64 matrix, P=100, 3K≤60 → ~48 KB). The memory risk lives in Tier-2 (model + feature cache), addressed when that tier is built. |
| Disk | 30 GB (tight) | Per-run metrics are written as **Parquet/CSV**, never pickled objects. No raw imagery is duplicated in Tier-1. A `clean` CLI path removes `results/`. Estimated disk usage is logged at the end of every experiment. |

## Defaults reflecting this target

- `n_workers: 8` in every config (override per machine).
- Outputs go to `results/` (git-ignored), as Parquet (falls back to CSV if `pyarrow` is unavailable).
- `python -m uavbench smoke` prints an **evals/sec** micro-benchmark and a **projected full-experiment runtime** so a paper run can be sized before launch.
- `python -m uavbench clean` deletes `results/`.

## Tier-1 footprint (this pass)

Repo + venv + Tier-1 results stay well under 1 GB. The instance "library" is regenerated
deterministically from seeds rather than stored, so there is no large on-disk instance set.

## Tier-2 footprint (deferred)

Tier-2 will reuse the existing `data_loader.py` real seismic dataset and cache **frozen vision
features in float16** keyed by building ID from a lightweight MobileNetV2/ResNet-18 backbone.
The cache size and RAM headroom will be documented when that tier is implemented.
