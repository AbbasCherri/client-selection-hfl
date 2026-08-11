#!/usr/bin/env python
"""Characterise the Noto 2024 dataset as the paper's setup section needs it.

Absent until 2026-08-11, which is a gap a reviewer would find immediately: every
claim here rests on a non-IID geospatial partition, and nothing in the repo said
how non-IID it actually is or how the classes are distributed.

Reports:
  * class distribution over the whole dataset, and the constant-predictor
    macro-F1 floor that follows from it (the gate's reference point)
  * per-client sample counts — min/median/max and Gini, i.e. quantity skew
  * per-client class composition — how many clients hold each class at all,
    and the mean per-client class entropy, i.e. label skew
  * geographic spread of the client coordinates

The class-imbalance and label-skew numbers are what justify macro-F1 as the
endpoint and explain why narrow UAV shards go single-class.

Usage:  python scripts/dataset_stats.py [--n 200] [--out results/dataset_stats]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/dataset_stats")
    ap.add_argument("--subsample", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from uavbench.analysis.collapse import constant_predictor_macro_f1
    from uavbench.fl.federated import _load_data
    from uavbench.metrics.fl import CLASS_NAMES

    cfg = {
        "data": {"source": "real", "subsample": args.subsample, "seed": 42,
                 "partition_seed": 0, "val_ratio": 0.1, "data_dir": "./data",
                 "feature_batch_size": 32, "N_clients": args.n},
        "fl": {"K": 1, "R_comm": 20000.0, "capacity": args.n},
    }
    full_dataset, client_train, test_idx, coords, _feat, val_idx = _load_data(cfg, out)
    labels = np.asarray(full_dataset.labels)

    # ---- class distribution + the gate's floor ----------------------------
    counts = np.array([(labels == i).sum() for i in range(len(CLASS_NAMES))])
    floor = constant_predictor_macro_f1(counts)
    cls = pd.DataFrame({
        "class": list(CLASS_NAMES),
        "count": counts,
        "share": (counts / counts.sum()).round(4),
    })
    print("=== class distribution ===")
    print(cls.to_string(index=False))
    print(f"\nmajority share {counts.max() / counts.sum():.4f}  "
          f"imbalance ratio {counts.max() / max(counts.min(), 1):.1f}:1")
    print(f"constant-predictor macro-F1 floor = {floor:.4f}")
    cls.to_csv(out / "class_distribution.csv", index=False)

    # ---- per-client quantity and label skew -------------------------------
    sizes, entropies, n_classes_held = [], [], []
    holds = np.zeros(len(CLASS_NAMES), dtype=int)
    for cid, idx in client_train.items():
        y = labels[np.asarray(idx)]
        sizes.append(len(y))
        c = np.array([(y == i).sum() for i in range(len(CLASS_NAMES))])
        holds += (c > 0).astype(int)
        n_classes_held.append(int((c > 0).sum()))
        p = c[c > 0] / c.sum() if c.sum() else np.array([1.0])
        h = -(p * np.log(p)).sum() / np.log(len(CLASS_NAMES))
        entropies.append(float(h))
    sizes = np.array(sizes)

    skew = pd.DataFrame([{
        "n_clients": len(sizes),
        "samples_min": int(sizes.min()),
        "samples_median": float(np.median(sizes)),
        "samples_max": int(sizes.max()),
        "samples_gini": round(gini(sizes), 4),
        "mean_client_class_entropy": round(float(np.mean(entropies)), 4),
        "clients_with_1_class": int((np.array(n_classes_held) == 1).sum()),
        "clients_with_all_classes": int(
            (np.array(n_classes_held) == len(CLASS_NAMES)).sum()),
    }])
    print("\n=== per-client skew ===")
    print(skew.to_string(index=False))
    skew.to_csv(out / "client_skew.csv", index=False)

    held = pd.DataFrame({"class": list(CLASS_NAMES), "clients_holding": holds,
                         "pct_of_clients": (holds / len(sizes) * 100).round(1)})
    print("\n=== how many clients hold each class at all ===")
    print(held.to_string(index=False))
    held.to_csv(out / "class_coverage_by_client.csv", index=False)

    # ---- geography --------------------------------------------------------
    if coords:
        ll = np.array(list(coords.values()), dtype=float)
        geo = pd.DataFrame([{
            "lat_min": ll[:, 0].min(), "lat_max": ll[:, 0].max(),
            "lon_min": ll[:, 1].min(), "lon_max": ll[:, 1].max(),
            "lat_span_km": round((ll[:, 0].max() - ll[:, 0].min()) * 111.0, 2),
            "lon_span_km": round(
                (ll[:, 1].max() - ll[:, 1].min()) * 111.0
                * float(np.cos(np.radians(ll[:, 0].mean()))), 2),
        }])
        print("\n=== client geography ===")
        print(geo.to_string(index=False))
        geo.to_csv(out / "geography.csv", index=False)

    print(f"\nsplit sizes: train {sum(len(v) for v in client_train.values())}, "
          f"val {len(val_idx)}, test {len(test_idx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
