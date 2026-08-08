#!/usr/bin/env python
"""Merge the four class-realism arms into one table the significance CLI can read.

Each arm is a separate ``run_sweep`` (they differ by ``fl.class_source`` and
``fl.placement_class_aware``, which are config-level knobs, not method-level
ones), so all four land in different directories with the *same* method name,
``proposed_hfl``. ``uavbench significance`` pairs on the ``method`` column, so
this rewrites that column to the arm name and concatenates into
``paper_sweep_rounds.parquet`` — a name ``_SIG_TABLES`` already recognises,
grouped by ``N``.

The pairing stays valid because every arm shares ``optimizer_seed``,
``data.seed``, ``N_values`` and ``methods`` — identical per-job seeds arm for
arm, which is exactly the condition ``_paired_values`` enforces.

Usage:  python scripts/merge_class_realism.py [--out results/class_realism]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavbench.reporting.tables import write_table  # noqa: E402

# arm directory -> the name it gets in the merged `method` column.
#
# Two groups, merged SEPARATELY and never pooled: they sit at different R_comm,
# and `_SIG_TABLES` groups paper_sweep_rounds by N alone (200 for every arm), so
# concatenating them would silently pair 20 km runs against 8 km runs as though
# they were the same condition.
#
#   main — R_comm 20 km, the paper operating point. Valid for the SELECTION
#          contrasts. Its placement contrast (true_placeaware vs
#          true_placeblind) is powerless: proposed_hfl already covers 99.95%
#          there, so class-aware placement has nothing left to steer.
#   bind — R_comm 8 km, where coverage is partial. This is the arm pair that
#          actually tests class-aware placement.
GROUPS = {
    "main": {
        "results/class_realism_true": "true_placeaware",
        "results/class_realism_pseudo": "pseudo_placeaware",
        "results/class_realism_noplace": "true_placeblind",
        "results/class_realism_none": "none_placeblind",
    },
    "bind": {
        "results/class_realism_bind": "bind_placeaware",
        "results/class_realism_bind_noplace": "bind_placeblind",
    },
}


def _read_arm(d: Path) -> pd.DataFrame:
    parquet, csv = d / "paper_sweep_rounds.parquet", d / "paper_sweep_rounds.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"{d}: no paper_sweep_rounds.parquet or .csv — did this arm run?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main", choices=sorted(GROUPS))
    ap.add_argument("--out", default=None, help="default: results/class_realism_<group>_merged")
    args = ap.parse_args()

    arms = GROUPS[args.group]
    out_rel = args.out or f"results/class_realism_{args.group}_merged"

    root = Path(__file__).resolve().parents[1]
    frames = []
    for rel, arm in arms.items():
        df = _read_arm(root / rel)
        got = set(df["method"].unique())
        if got != {"proposed_hfl"}:
            raise ValueError(
                f"{rel}: expected only proposed_hfl, got {sorted(got)} — the arms "
                "must differ by config alone, or renaming the method column would "
                "collapse distinct methods into one arm"
            )
        df = df.copy()
        df["method"] = arm
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out = root / out_rel
    out.mkdir(parents=True, exist_ok=True)
    path = write_table(merged, out / "paper_sweep_rounds.parquet")
    print(f"[{args.group}] merged {len(frames)} arms, {len(merged)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
