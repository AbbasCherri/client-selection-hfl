#!/usr/bin/env python
"""Fail a Tier-1 run whose vertical search collapsed back to a bound.

The Tier-1 counterpart of scripts/gate_collapse.py, and it exists for the same
reason: a degenerate run produces complete, well-formed, entirely plausible
output. Until 2026-08-09 Tier-1 scored coverage through a flat
``slant_distance <= R_comm`` sphere, under which altitude can only ever ADD
slant distance — so every optimizer drove z to the floor and the benchmark
advertised a 3D search whose third dimension had a closed-form answer. Nothing
in the tables showed it.

What is checked, and what is deliberately NOT
---------------------------------------------
A single method sitting on a bound is a legitimate result about that method:
Mozaffari-2016 flies one packing-derived altitude, and a rule that genuinely
prefers the floor should be free to say so. So this does not fail per method.

It fails when EVERY method pins to the same bound, which is not a finding about
any of them — it is the signature of a coverage model in which altitude cannot
pay for itself, i.e. the link model is not reaching the scorer.

Usage:
    python scripts/gate_altitude.py results/tier1_core --config configs/tier1_core.yaml

Exit 0 if the vertical decision is live, 1 if it has gone degenerate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavbench.runner import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--config", required=True, help="config the run used (for the band)")
    ap.add_argument(
        "--margin-frac",
        type=float,
        default=0.02,
        help="fraction of the band's height counted as 'on the bound'",
    )
    args = ap.parse_args()

    d = Path(args.results_dir)
    runs = d / "runs.parquet"
    if not runs.exists():
        print(f"[altitude] {d}: no runs.parquet — cannot assess")
        return 1
    df = pd.read_parquet(runs)

    if "mean_altitude_m" not in df.columns:
        print(
            f"[altitude] {d}: no mean_altitude_m column — this run predates altitude "
            "reporting, so its vertical behaviour cannot be verified. Re-run it."
        )
        return 1

    cfg = load_config(args.config)
    z_lo, z_hi = (float(v) for v in cfg["area"]["z"])
    margin = args.margin_frac * (z_hi - z_lo)

    per_method = df.groupby("method")["mean_altitude_m"].mean().sort_values()
    pinned = {}
    for method, z in per_method.items():
        at_floor = z <= z_lo + margin
        at_ceil = z >= z_hi - margin
        flag = "floor" if at_floor else ("ceiling" if at_ceil else "")
        if flag:
            pinned[method] = flag
        print(f"[altitude] {method:<22} mean z = {z:8.1f} m  {('<- ' + flag) if flag else ''}")

    print(f"[altitude] band [{z_lo:.0f}, {z_hi:.0f}] m, margin {margin:.1f} m")

    if len(pinned) == len(per_method) and len(set(pinned.values())) == 1:
        bound = next(iter(pinned.values()))
        print(
            f"[altitude] DEGENERATE: all {len(pinned)} methods pinned to the {bound}. "
            "Altitude is not a decision variable in this run — check that "
            "problem.link_model is 'path_loss' and that R_comm lies inside the "
            "band's coherent interval (z/tan(theta_opt))."
        )
        return 1

    if pinned:
        print(
            f"[altitude] note: {len(pinned)}/{len(per_method)} methods sit on a bound "
            "— legitimate per method (Mozaffari flies one packing altitude), reported "
            "for visibility, not a failure."
        )
    print("[altitude] vertical decision is live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
