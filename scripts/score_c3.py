#!/usr/bin/env python
"""Score the C3 arm against REPORTS/preregistration_v6_c3.md §4.

§4 was committed (c02625c9) before any C3 code existed and its criteria are NOT
the same as v6's — C3 is judged on how much of the `moon2022` gap it closes at
the two coverage-saturated identification points, not on beating `moon2022`
outright. Applied mechanically here so the verdict cannot become a matter of
which comparison gets quoted afterwards.

The mechanism is identified only if ALL of:
  1. closes >= half the moon2022 - mclp_place gap at BOTH K=20 and K=30.
     Gaps measured in v5 are +0.035 and +0.039, so C3 must reach >= +0.018 and
     >= +0.020 over mclp_place at those K.
  2. is not negative against mclp_place at any K in {5,10,15,20,30}
  3. does not lose to flat_fl at K=20 (above it, or a CI containing zero)
  4. no cell it wins in is degenerate

Partial success is reported as partial. It is not rounded up.

Endpoint: macro-F1, mean of the last 10 rounds per seed, paired Wilcoxon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uavbench.analysis.collapse import check_not_collapsed  # noqa: E402

BASE = Path("results/paper_uav_count")
ARM = "v6_c3_disjoint"
KS = (5, 10, 15, 20, 30)
ID_POINTS = (20, 30)          # named in §5a, before any C3 result existed
F1_COLS = ("f1_survived", "f1_collapsed", "f1_obstructed", "f1_missing")
CLASS_COUNTS = np.array([0.8147, 0.0264, 0.0648, 0.0941]) * 1e4


def _last10(df: pd.DataFrame) -> pd.DataFrame:
    tail = df[df["round"] > df["round"].max() - 10]
    return tail.groupby(["seed"], as_index=False).mean(numeric_only=True)


def _paired(a: pd.Series, b: pd.Series):
    shared = sorted(set(a.index) & set(b.index))
    x, y = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
    if np.allclose(x - y, 0):
        return 0.0, 1.0
    _s, p = wilcoxon(x, y)
    return float((x - y).mean()), float(p)


def _series(df: pd.DataFrame, k: int) -> pd.Series:
    return _last10(df[df["K"] == k]).set_index("seed")["macro_f1"]


def main() -> int:
    base = pd.read_parquet(BASE / "uav_sweep_rounds.parquet")
    ref = {m: base[base["method"] == m] for m in ("moon2022", "mclp_place", "flat_fl")}

    arm_pq = sorted(Path("results", ARM).glob("*.parquet"))
    if not arm_pq:
        print(f"!! results/{ARM} missing — run the arm first")
        return 1
    adf = pd.read_parquet(arm_pq[0])

    print(f"### {ARM}")
    print(f"{'K':>4} {'C3':>8} {'mclp':>8} {'diff':>8} {'p':>8} | "
          f"{'moon':>8} {'gap':>8} {'closed':>8}")

    diffs, closed, degenerate = {}, {}, []
    for k in KS:
        a = _series(adf, k)
        m = _series(ref["mclp_place"], k)
        mo = _series(ref["moon2022"], k)
        d, p = _paired(a, m)
        gap = mo.mean() - m.mean()
        frac = d / gap if abs(gap) > 1e-9 else np.nan
        diffs[k], closed[k] = d, frac
        print(f"{k:>4} {a.mean():>8.4f} {m.mean():>8.4f} {d:>+8.4f} {p:>8.4f} | "
              f"{mo.mean():>8.4f} {gap:>+8.4f} {frac:>7.0%}")

        cell = adf[adf["K"] == k]
        cell = cell[cell["round"] > cell["round"].max() - 10]
        per_class = {c: float(cell[c].mean()) for c in F1_COLS if c in cell}
        if not check_not_collapsed(float(cell["macro_f1"].mean()), per_class,
                                   CLASS_COUNTS).ok:
            degenerate.append(k)

    c1 = all(np.isfinite(closed[k]) and closed[k] >= 0.5 for k in ID_POINTS)
    c2 = all(diffs[k] >= 0 for k in KS)
    d20, p20 = _paired(_series(adf, 20), _series(ref["flat_fl"], 20))
    c3 = d20 > 0 or p20 >= 0.05
    c4 = not degenerate

    print()
    print(f"  [1] closes >=50% of the moon2022 gap at K=20 AND 30 : {c1}  "
          f"(K=20 {closed[20]:.0%}, K=30 {closed[30]:.0%})")
    print(f"  [2] never negative vs mclp_place                    : {c2}  "
          f"(min {min(diffs.values()):+.4f} at K={min(diffs, key=diffs.get)})")
    print(f"  [3] not losing to flat_fl at K=20                   : {c3}  "
          f"(diff {d20:+.4f}, p={p20:.4f})")
    print(f"  [4] no degenerate cell                              : {c4}  "
          f"{'(degenerate at K=' + str(degenerate) + ')' if degenerate else ''}")

    print("\n" + "=" * 62)
    if c1 and c2 and c3 and c4:
        print("VERDICT: C3 MEETS every pre-registered criterion — redundant overlap")
        print("is identified as the mechanism behind moon2022's advantage.")
        return 0
    # Say plainly which parts held, without letting a partial read as a pass.
    part = [n for n, ok in ((1, c1), (2, c2), (3, c3), (4, c4)) if ok]
    print("VERDICT: C3 does NOT meet the pre-registered criteria.")
    print(f"Criteria met: {part or 'none'}. Report as a partial result at most —")
    print("§4 says partial success is reported as partial, not rounded up.")
    if any(np.isfinite(closed[k]) and closed[k] >= 0.5 for k in ID_POINTS):
        print("Note: the gap DID close at one identification point but not both;")
        print("that is the partial case §4 anticipated, not a win.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
