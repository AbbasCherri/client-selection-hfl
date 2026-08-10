#!/usr/bin/env python
"""Score the C2 power run against REPORTS/preregistration_c2_power.md §4.

Three criteria, all required:
  1. Holm-significant wins over mclp_place at >= 2 of the 5 fleet sizes, grid (a)
  2. positive sign at all 5 in grid (a)
  3. positive sign in >= 3 of the 4 cells of grid (b) — the effect is not
     confined to the single client count C2 was discovered at

Also printed, and NOT decisive: the independent subset (seeds 10-24 of grid a),
which is underpowered on its own and is shown for transparency only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uavbench.analysis.collapse import check_not_collapsed  # noqa: E402

KS = (5, 10, 15, 20, 30)
F1_COLS = ("f1_survived", "f1_collapsed", "f1_obstructed", "f1_missing")
CLASS_COUNTS = np.array([0.8147, 0.0264, 0.0648, 0.0941]) * 1e4


def last10(df: pd.DataFrame) -> pd.Series:
    tail = df[df["round"] > df["round"].max() - 10]
    return tail.groupby("seed")["macro_f1"].mean()


def load(stem: str) -> pd.DataFrame | None:
    pq = sorted(Path("results", stem).glob("*.parquet"))
    return pd.read_parquet(pq[0]) if pq else None


def paired(a: pd.Series, b: pd.Series):
    idx = sorted(set(a.index) & set(b.index))
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    if len(idx) < 3 or np.allclose(x, y):
        return 0.0, 1.0, len(idx)
    return float((x - y).mean()), float(wilcoxon(x, y).pvalue), len(idx)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        prev = max(prev, p * (m - i))
        out[k] = prev < 0.05
    return out


def main() -> int:
    c2, base = load("c2_power_c2"), load("c2_power_base")
    if c2 is None or base is None:
        print("!! grid (a) results missing — run scripts/run_c2_power.sh first")
        return 1

    print("### grid (a) — fleet replication, N=200, 25 seeds")
    print(f"{'K':>4} {'C2':>8} {'base':>8} {'diff':>8} {'p':>8} {'n':>4}")
    diffs, pv, degenerate = {}, {}, []
    for k in KS:
        a, b = last10(c2[c2["K"] == k]), last10(base[base["K"] == k])
        d, p, n = paired(a, b)
        diffs[k], pv[k] = d, p
        print(f"{k:>4} {a.mean():>8.4f} {b.mean():>8.4f} {d:>+8.4f} {p:>8.4f} {n:>4}")
        cell = c2[c2["K"] == k]
        cell = cell[cell["round"] > cell["round"].max() - 10]
        pc = {c: float(cell[c].mean()) for c in F1_COLS if c in cell}
        if not check_not_collapsed(float(cell["macro_f1"].mean()), pc, CLASS_COUNTS).ok:
            degenerate.append(k)

    sig = holm(pv)
    c1 = sum(sig.values()) >= 2
    c2ok = all(diffs[k] > 0 for k in KS)

    print("\n### grid (b) — client-count generalisation, 25 seeds")
    print(f"{'N':>4} {'K':>4} {'C2':>8} {'base':>8} {'diff':>8} {'p':>8}")
    bpos = 0
    bcells = 0
    for n_clients in (50, 100):
        gc = load(f"c2_power_n{n_clients}_c2")
        gb = load(f"c2_power_n{n_clients}_base")
        if gc is None or gb is None:
            print(f"{n_clients:>4}    ?  (missing)")
            continue
        for k in (10, 20):
            a, b = last10(gc[gc["K"] == k]), last10(gb[gb["K"] == k])
            d, p, _ = paired(a, b)
            bcells += 1
            bpos += d > 0
            print(f"{n_clients:>4} {k:>4} {a.mean():>8.4f} {b.mean():>8.4f} "
                  f"{d:>+8.4f} {p:>8.4f}")
    c3 = bpos >= 3 and bcells == 4

    print("\n### independent subset (grid a, seeds 10-24) — NOT decisive")
    for k in KS:
        a = last10(c2[(c2["K"] == k) & (c2["seed"] >= 10)])
        b = last10(base[(base["K"] == k) & (base["seed"] >= 10)])
        d, p, n = paired(a, b)
        print(f"  K={k:<3} diff {d:+.4f}  p={p:.4f}  n={n}")

    print()
    print(f"  [1] Holm-sig wins at >=2 of 5 fleet sizes : {c1}  "
          f"(sig at {[k for k, s in sig.items() if s and diffs[k] > 0]})")
    print(f"  [2] positive at all 5 fleet sizes         : {c2ok}  "
          f"(min {min(diffs.values()):+.4f} at K={min(diffs, key=diffs.get)})")
    print(f"  [3] positive in >=3 of 4 grid-(b) cells   : {c3}  ({bpos}/{bcells})")
    print(f"  [-] no degenerate cell                    : {not degenerate}  "
          f"{degenerate or ''}")

    print("\n" + "=" * 62)
    if c1 and c2ok and c3 and not degenerate:
        print("VERDICT: C2 is a real effect on the pre-registered criteria.")
        print("It improves the PROPOSED PLACEMENT's own numbers. It does not beat")
        print("moon2022, does not rescue any selection claim, and does not reopen")
        print("the v6 verdict — see §6 of the pre-registration.")
        return 0
    if c1 and c2ok and not c3:
        print("VERDICT: PARTIAL. C2 replicates at N=200 but does not generalise")
        print("across client count on this evidence. Report exactly that.")
        return 1
    print("VERDICT: C2 does NOT meet the pre-registered criteria.")
    print("Report the negative result; §6 says failure is reportable.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
