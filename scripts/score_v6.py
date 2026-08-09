#!/usr/bin/env python
"""Score the v6 arms against the criteria fixed in the pre-registration.

REPORTS/preregistration_v6_method.md was committed (9729a1ce) before any v6 code
existed. This script applies its §4 mechanically so the verdict is not a matter
of which comparison gets quoted afterwards.

An improvement is declared only if ALL of:
  1. beats moon2022 at K=10, 15, 20 — Holm-significant at >= 2 of the 3
  2. beats mclp_place at every K in {5, 10, 15, 20, 30}
  3. does not lose to flat_fl at K=20 (above it, or a CI containing zero)
  4. no cell it wins in is degenerate

Endpoint: macro-F1, mean of the last 10 rounds per seed, paired Wilcoxon,
Holm within fleet size. Prints a per-criterion verdict and a final line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uavbench.analysis.collapse import (  # noqa: E402
    check_not_collapsed,
    constant_predictor_macro_f1,
)

BASE = Path("results/paper_uav_count")
ARMS = ("v6_c1_reachable", "v6_c2_diversity", "v6_both")
PRIMARY = "v6_both"
KS = (5, 10, 15, 20, 30)
F1_COLS = ("f1_survived", "f1_collapsed", "f1_obstructed", "f1_missing")


def _last10(df: pd.DataFrame, by=("seed",)) -> pd.DataFrame:
    tail = df[df["round"] > df["round"].max() - 10]
    return tail.groupby(list(by), as_index=False).mean(numeric_only=True)


def _holm(pvals: dict[str, float]) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, p * (m - i))
        out[k] = adj < 0.05
        prev = adj
    return out


def _paired(a: pd.Series, b: pd.Series):
    shared = sorted(set(a.index) & set(b.index))
    x, y = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
    diff = x - y
    if np.allclose(diff, 0):
        return 0.0, 1.0, len(shared)
    _stat, p = wilcoxon(x, y)
    return float(diff.mean()), float(p), len(shared)


def _series(df: pd.DataFrame, k: int) -> pd.Series:
    sub = _last10(df[df["K"] == k]) if "K" in df.columns else _last10(df)
    return sub.set_index("seed")["macro_f1"]


def main() -> int:
    base = pd.read_parquet(BASE / "uav_sweep_rounds.parquet")
    floor = None
    confs = sorted(BASE.rglob("confusion.parquet"))
    if confs:
        c = pd.concat([pd.read_parquet(p) for p in confs], ignore_index=True)
        c = c[c["round"] == c["round"].max()]
        floor = constant_predictor_macro_f1(c.groupby("true_label")["count"].sum().to_numpy())

    ref = {m: base[base["method"] == m] for m in ("moon2022", "mclp_place", "flat_fl")}

    verdicts = {}
    for arm in ARMS:
        p = Path("results") / arm / "uav_sweep_rounds.parquet"
        if not p.exists():
            print(f"\n### {arm}: NOT RUN")
            continue
        adf = pd.read_parquet(p)
        print(f"\n### {arm}")
        print(f"{'K':>4} {'arm':>8} {'moon2022':>9} {'diff':>8} {'p':>8} | "
              f"{'mclp':>8} {'diff':>8} {'p':>8}")

        pm, pc, diffs_moon, diffs_mclp, degenerate = {}, {}, {}, {}, []
        for k in KS:
            a = _series(adf, k)
            dm, pmv, _ = _paired(a, _series(ref["moon2022"], k))
            dc, pcv, _ = _paired(a, _series(ref["mclp_place"], k))
            pm[k], pc[k] = pmv, pcv
            diffs_moon[k], diffs_mclp[k] = dm, dc
            print(f"{k:>4} {a.mean():>8.4f} {_series(ref['moon2022'], k).mean():>9.4f} "
                  f"{dm:>+8.4f} {pmv:>8.4f} | {_series(ref['mclp_place'], k).mean():>8.4f} "
                  f"{dc:>+8.4f} {pcv:>8.4f}")

            if floor is not None:
                cell = _last10(adf[adf["K"] == k])
                per_class = {c: float(cell[c].mean()) for c in F1_COLS if c in cell}
                v = check_not_collapsed(float(cell["macro_f1"].mean()), per_class,
                                        np.array([0.8147, 0.0264, 0.0648, 0.0941]) * 1e4)
                if not v.ok:
                    degenerate.append(k)

        sig_moon = _holm({k: pm[k] for k in (10, 15, 20)})
        c1 = sum(sig_moon.values()) >= 2 and all(diffs_moon[k] > 0 for k in (10, 15, 20))
        c2 = all(diffs_mclp[k] > 0 for k in KS)
        df20, p20, _ = _paired(_series(adf, 20), _series(ref["flat_fl"], 20))
        c3 = df20 > 0 or p20 >= 0.05
        c4 = not degenerate

        print(f"  [1] beats moon2022 at K=10/15/20, Holm-sig >=2 : {c1}  "
              f"(sig at {[k for k, s in sig_moon.items() if s]})")
        print(f"  [2] beats mclp_place at every K                : {c2}")
        print(f"  [3] not losing to flat_fl at K=20              : {c3}  "
              f"(diff {df20:+.4f}, p={p20:.4f})")
        print(f"  [4] no degenerate cell                         : {c4}  "
              f"{'(degenerate at K=' + str(degenerate) + ')' if degenerate else ''}")
        verdicts[arm] = c1 and c2 and c3 and c4

    print("\n" + "=" * 62)
    if PRIMARY not in verdicts:
        print(f"VERDICT: {PRIMARY} did not run — no verdict.")
        return 1
    if verdicts[PRIMARY]:
        print(f"VERDICT: {PRIMARY} MEETS every pre-registered criterion.")
        return 0
    print(f"VERDICT: {PRIMARY} does NOT meet the pre-registered criteria.")
    print("Report the negative result. The diagnosis it rests on — a capacity-capped")
    print("placement objective is mis-specified for multi-round FL — stands either way.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
