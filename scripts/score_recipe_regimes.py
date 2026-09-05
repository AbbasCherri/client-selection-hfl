"""Recipe-regime triangulation: does the method ordering depend on the recipes?

Three regimes, same architecture (client_full), same reduction (mean macro-F1
over the last 10 rounds), paired Wilcoxon:

  A shared      results/paper_coverage_cf_mclp  every method on proposed_hfl's recipe
  B per-method  results/paper_coverage_final    each method on its own 2026-08-05 recipe
  C alt         results/robustness_alt_recipes  P1's 5 km recipes, seeds 20-24

A and B sweep R_comm at N=200; C sweeps N at R=5000. They are not the same cells,
so the comparison is of SIGN and ORDER, not of magnitude.

C has n=5, where the minimum two-sided Wilcoxon p is 0.0625. Nothing in C can be
significant by construction; it is a sign check and is reported as one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

COMPARATORS = ("fedcs", "oort", "flat_fl", "hfl_no_selection")


def load(stem):
    for name in ("paper_sweep_rounds.parquet", "coverage_sweep_rounds.parquet"):
        p = Path("results", stem, name)
        if p.exists():
            return pd.read_parquet(p)
    q = sorted(Path("results", stem).glob("*.parquet"))
    return pd.read_parquet(q[0]) if q else None


def last10(df, method, col, val):
    sub = df[(df["method"] == method) & (np.isclose(df[col], val))]
    if sub.empty:
        return None
    tail = sub[sub["round"] > sub["round"].max() - 10]
    return tail.groupby("seed")["macro_f1"].mean()


def paired(a, b):
    if a is None or b is None:
        return float("nan"), 1.0, 0
    idx = sorted(set(a.index) & set(b.index))
    if len(idx) < 3:
        return float("nan"), 1.0, len(idx)
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    if np.allclose(x, y):
        return 0.0, 1.0, len(idx)
    return float((x - y).mean()), float(wilcoxon(x, y).pvalue), len(idx)


def block(title, df, col, vals, note=""):
    print(f"\n=== {title} ===")
    if note:
        print(f"    {note}")
    present = [c for c in COMPARATORS if c in set(df["method"])]
    head = "  ".join(f"{col}={v:g}".rjust(11) for v in vals)
    print(f"{'proposed_hfl minus':>20}  {head}")
    signs = {}
    for comp in present:
        cells, sg = [], []
        for v in vals:
            d, p, k = paired(last10(df, "proposed_hfl", col, v),
                             last10(df, comp, col, v))
            cells.append(("  n/a" if np.isnan(d)
                          else format(d, "+.4f") + ("*" if p < 0.05 else " ")))
            sg.append(0 if np.isnan(d) else int(np.sign(d)))
        signs[comp] = sg
        print(f"{comp:>20}  " + "  ".join(c.rjust(11) for c in cells))
    return signs


def main() -> int:
    a = load("paper_coverage_cf_mclp")
    b = load("paper_coverage_final")
    c = load("robustness_alt_recipes")

    print("# recipe-regime triangulation — does the ordering depend on the recipes?")
    print("# reduction: mean macro-F1 over last 10 rounds; * = raw Wilcoxon p < 0.05")

    radii = sorted(b["R_comm"].unique()) if b is not None else []
    sa = block("REGIME A - shared recipe (proposed_hfl's), R_comm sweep, n=10",
               a, "R_comm", radii) if a is not None else {}
    sb = block("REGIME B - per-method 2026-08-05 recipes, R_comm sweep, n=10",
               b, "R_comm", radii) if b is not None else {}
    if c is not None:
        ns = sorted(c["N"].unique())
        seeds = sorted(c["seed"].unique())
        block("REGIME C - P1 5km recipes, N sweep, n=%d" % len(seeds),
              c, "N", ns,
              note=("seeds %s; min two-sided Wilcoxon p at n=%d is %.4f, so no cell "
                    "here can reach 0.05 - sign check only"
                    % (seeds, len(seeds), 2.0 ** -(len(seeds) - 1))))
        tail = c[c["round"] > c["round"].max() - 10]
        print("\n    ranking by mean macro-F1:")
        for n in ns:
            r = (tail[tail["N"] == n].groupby("method")["macro_f1"]
                 .mean().sort_values(ascending=False))
            print(f"      N={n:<4} " + "  ".join(f"{m}={v:.4f}" for m, v in r.items()))

    if sa and sb:
        print("\n=== sign agreement between regimes A and B (same cells) ===")
        for comp in sa:
            if comp not in sb:
                continue
            agree = sum(int(x == y and x != 0) for x, y in zip(sa[comp], sb[comp]))
            tot = sum(int(x != 0 and y != 0) for x, y in zip(sa[comp], sb[comp]))
            flag = "" if agree == tot else "   <-- ORDERING IS RECIPE-DEPENDENT"
            print(f"  vs {comp:<18} {agree}/{tot} radii agree{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
