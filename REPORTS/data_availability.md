# Data Availability Statement

## Real-world dataset (single event)

All real-data experiments use one fused, geo-located multi-modal dataset for
the **2024 Noto Peninsula earthquake** (Japan), streamed from HuggingFace:

- **Dataset**: `AbbasABC/HFL-Dataset`
- **Pinned revision**: `6cf97c900445e080e61cb45e1aa72515d3ff1de8`
  (default in `src/hflsim/data/loader.py`; overridable via the
  `HF_DATASET_REVISION` environment variable — every GCP wrapper script pins
  the same hash)
- **Subsample fraction**: per config — `configs/paper_full.yaml` uses
  `subsample: 1.0` (~128k rows); reduced configs use 0.05 for smoke tests.
- **Modalities fused per building**:
  1. Building damage assessment labels from the Noto Peninsula 2024
     post-event survey (raw codes {0 Survived, 1 Collapsed, 9 Obstructed
     view, 99 Missing/inconsistent}, remapped to contiguous {0..3});
  2. USGS ShakeMap seismic intensity parameters (MMI, PGA, PGV, SA 0.3/1.0/3.0 s);
  3. On-demand aerial imagery chips from the GSI (Geospatial Information
     Authority of Japan) XYZ tile service, zoom 18 (~0.6 m/px), cached
     locally under `data/tile_cache/`.

### Licensing and redistribution

- The GSI aerial tiles are used under the **Government of Japan Standard
  Terms of Use v2.0**; cached tiles are **not redistributed** with this
  repository — they are re-fetched on demand from the public GSI service.
- Any use of the HF dataset must respect the upstream terms of the fused
  sources; the pinned revision hash above makes the exact evaluated
  snapshot citable and re-fetchable.
- Reruns require a HuggingFace token (`HF_TOKEN`) for streaming; the
  metadata/partition caches under `data/` are derived artifacts and are
  regenerated automatically.

### Data-quality diagnostic

The measured **black-chip rate** (fraction of image loads that fall back to
an all-black chip after GSI fetch failure) is logged and persisted per run
under `_diagnostics.black_chip_rate` in each harness's resolved-config YAML.
Report this number alongside accuracy results: a high rate means the image
modality carried no signal (see `REPORTS/full_system_implementation_details.md`).

## Stated limitation: single-event scope

Every real-data result in the paper is derived from this **one** seismic
event. No second real, geo-located, multi-modal damage dataset with
ShakeMap-equivalent parameters and public post-event imagery was available
at the time of writing, and claims should be scoped accordingly: the
real-data evaluation demonstrates feasibility and comparative method
behaviour **on this event**, not cross-event generalization.

As the controlled complement, the **synthetic stress-test sweep**
(`configs/synthetic_stress_test.yaml`, `uavbench run_stress_sweep`) varies
degradation axes the single event does not exhibit on demand — per-round
device dropout, aftershock-triggered area-wide SNR degradation, and
black-chip (unusable imagery) rate — providing robustness evidence under
conditions beyond the recorded event. See `src/uavbench/fl/stress_sweep.py`
for the paired seeding design.

## Synthetic data

`SyntheticClientData` (`src/uavbench/fl/dataset.py`) generates deterministic
offline data: Noto-region coordinates, Gaussian seismic features, the
empirical class imbalance (60/20/10/10), and Gaussian image features. Its
`black_chip_rate` knob zeroes a fraction of image-feature rows — an
artificial proxy for the effect of fetch-failure chips, not a simulation of
the failure mechanism itself.
