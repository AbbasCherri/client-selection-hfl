#!/usr/bin/env python
"""Fail a finished run that collapsed onto the majority class.

Wired into the sweep runners so a degenerate block stops the pipeline instead of
feeding plausible-looking numbers into three days of downstream compute. On
2026-08-08 three configurations produced complete, well-formed, fully
checkpointed results in which the model had learned nothing but the majority
class, and nothing objected — see src/uavbench/analysis/collapse.py.

Class counts come from the run's OWN confusion matrix (true_label totals), not
from a hard-coded prior: a run at a different subsample or split has a different
label distribution, and calibrating the baseline against the wrong one would
either mask a collapse or invent one.

Usage:
    python scripts/gate_collapse.py results/class_realism_true [--method proposed_hfl]

Exit 0 if every method in the run cleared the gate, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavbench.analysis.collapse import check_not_collapsed  # noqa: E402

_ROUND_TABLES = (
    "paper_sweep_rounds.parquet",
    "uav_sweep_rounds.parquet",
    "coverage_sweep_rounds.parquet",
    "sweep_rounds.parquet",
    "fullsim_rounds.parquet",
)


def _load(results_dir: Path) -> pd.DataFrame:
    for name in _ROUND_TABLES:
        p = results_dir / name
        if p.exists():
            return pd.read_parquet(p)
        if p.with_suffix(".csv").exists():
            return pd.read_csv(p.with_suffix(".csv"))
    raise FileNotFoundError(f"no recognised round table in {results_dir}")


def _class_counts(results_dir: Path) -> pd.Series | None:
    """True-label totals from any confusion matrix under this run."""
    frames = []
    for p in sorted(results_dir.rglob("confusion.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:  # a corrupt shard must not veto the gate
            continue
    if not frames:
        return None
    conf = pd.concat(frames, ignore_index=True)
    last = conf[conf["round"] == conf["round"].max()]
    return last.groupby("true_label")["count"].sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--method", default=None, help="check only this method")
    ap.add_argument("--min-margin", type=float, default=0.05)
    ap.add_argument("--min-class-f1", type=float, default=0.05)
    args = ap.parse_args()

    d = Path(args.results_dir)
    df = _load(d)
    counts = _class_counts(d)
    if counts is None:
        print(f"[gate] {d}: no confusion matrix — cannot derive the baseline, refusing to pass")
        return 1

    f1_cols = [c for c in df.columns if c.startswith("f1_") and not c.startswith("f1_val")]
    if not f1_cols:
        print(f"[gate] {d}: no per-class F1 columns — cannot assess collapse")
        return 1

    # Final round per (method, seed), then averaged: a single seed's last round
    # is noisy enough to flip a borderline verdict either way.
    keys = [k for k in ("method", "seed") if k in df.columns]
    last = df.sort_values("round").groupby(keys, as_index=False).last()

    failed = []
    for method, sub in last.groupby("method"):
        if args.method and method != args.method:
            continue
        macro = float(sub["macro_f1"].mean())
        per_class = {c: float(sub[c].mean()) for c in f1_cols}
        v = check_not_collapsed(
            macro, per_class, counts.to_numpy(),
            min_margin=args.min_margin, min_class_f1=args.min_class_f1,
        )
        print(f"[gate] {method:<22} {v}")
        if not v.ok:
            failed.append(method)

    if failed:
        print(f"[gate] DEGENERATE: {', '.join(failed)} — this is not a usable result")
        return 1
    print("[gate] all methods cleared the collapse gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
