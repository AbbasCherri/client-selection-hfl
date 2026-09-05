"""Score the final radius sweep: proposed_hfl against each comparator across R_comm.

Reduction is the mean of the last 10 rounds per seed, matching every other
scorer in this project. Paired Wilcoxon across the 10 seeds; Holm correction
applied WITHIN each comparator family (the 6 radii), because the question asked
of each family is "does the sign/size depend on reach", not "is any single
radius special".

Prints the v5 curve beside it when results/paper_coverage_v5 is present. That
curve was measured under `fusion_owner: uav`; this one under `client_full`.
The whole point of the regeneration is whether the crossover survives.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MDE_N10 = 0.034
COMPARATORS = ("flat_fl", "hfl_no_selection", "fedcs", "oort")


def load(stem: str) -> pd.DataFrame | None:
    p = Path("results", stem, "coverage_sweep_rounds.parquet")
    if p.exists():
        return pd.read_parquet(p)
    pq = sorted(Path("results", stem).glob("*.parquet"))
    return pd.read_parquet(pq[0]) if pq else None


def last10(df, method, r):
    sub = df[(df["method"] == method) & (np.isclose(df["R_comm"], r))]
    if sub.empty:
        return pd.Series(dtype=float)
    tail = sub[sub["round"] > sub["round"].max() - 10]
    return tail.groupby("seed")["macro_f1"].mean()


def paired(a, b):
    idx = sorted(set(a.index) & set(b.index))
    if len(idx) < 3:
        return float("nan"), 1.0, len(idx)
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    if np.allclose(x, y):
        return 0.0, 1.0, len(idx)
    return float((x - y).mean()), float(wilcoxon(x, y).pvalue), len(idx)


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        prev = max(prev, p * (m - i))
        out[k] = prev < 0.05
    return out


def main() -> int:
    stem = sys.argv[1] if len(sys.argv) > 1 else "paper_coverage_final"
    df = load(stem)
    if df is None:
        print(f"no results at results/{stem}")
        return 1
    radii = sorted(df["R_comm"].unique())
    old = load("paper_coverage_v5")

    print(f"# radius sweep verdict — {stem}")
    print(f"# {len(df)} rows, radii {radii}, methods {sorted(df['method'].unique())}")
    print(f"# reduction: mean macro_f1 over last 10 rounds; paired Wilcoxon over seeds;")
    print(f"# Holm applied within each comparator family ({len(radii)} radii)")

    summary = {}
    for comp in COMPARATORS:
        if comp not in set(df["method"]):
            continue
        diffs, pvals, npairs = {}, {}, {}
        for r in radii:
            d, p, k = paired(last10(df, "proposed_hfl", r), last10(df, comp, r))
            diffs[r], pvals[r], npairs[r] = d, p, k
        sig = holm(pvals)
        summary[comp] = (diffs, sig)
        print(f"\n=== proposed_hfl - {comp} ===")
        print(f"{'R_comm':>8}  {'diff':>9}  {'p':>9}  {'Holm':>6}  {'n':>3}  note")
        for r in radii:
            note = ""
            if not np.isnan(diffs[r]) and abs(diffs[r]) < MDE_N10 and not sig[r]:
                note = "below MDE - uninformative"
            print(f"{r:>8.0f}  {diffs[r]:>+9.4f}  {pvals[r]:>9.4f}  "
                  f"{str(sig[r]):>6}  {npairs[r]:>3}  {note}")
        if old is not None and comp in set(old["method"]):
            print(f"  v5 (fusion_owner=uav) for reference:")
            for r in radii:
                if not np.isclose(old['R_comm'], r).any():
                    continue
                d, p, k = paired(last10(old, "proposed_hfl", r), last10(old, comp, r))
                print(f"{r:>8.0f}  {d:>+9.4f}  {p:>9.4f}  {'':>6}  {k:>3}  v5")

    # crossover: lowest radius at which the sign of proposed - flat_fl turns non-negative
    if "flat_fl" in summary:
        diffs, sig = summary["flat_fl"]
        neg_sig = [r for r in radii if sig[r] and diffs[r] < 0]
        pos_sig = [r for r in radii if sig[r] and diffs[r] > 0]
        cross = next((r for r in radii if diffs[r] >= 0), None)
        print("\n=== crossover vs flat_fl ===")
        print(f"  sign flips non-negative at R_comm = {cross}")
        print(f"  Holm-significant LOSSES at : {[int(r) for r in neg_sig] or 'none'}")
        print(f"  Holm-significant WINS at   : {[int(r) for r in pos_sig] or 'none'}")
        mono = all(diffs[radii[i]] <= diffs[radii[i + 1]] + 1e-9
                   for i in range(len(radii) - 1))
        print(f"  monotone increasing in R_comm: {mono}")

    print("\n=== mechanism (mean over seeds, last 10 rounds) ===")
    tail = df[df["round"] > df["round"].max() - 10]
    cols = [c for c in ("coverage_pct", "mean_shard_clients", "participation_pct",
                        "n_unique_selected") if c in df.columns]
    piv = (tail[tail["method"] == "proposed_hfl"]
           .groupby("R_comm")[cols].mean().round(3))
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
