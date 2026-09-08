"""Is mean_shard_clients a sufficient mediator of the flat_fl deficit?

The N axis (paper_full_cf60) and the R_comm axis (paper_coverage_cf60) both move
clients-per-shard. If shard width is THE mechanism, cells with equal
mean_shard_clients should show equal deficit regardless of which axis produced
them. This prints both axes side by side so that can be checked rather than
assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def l10(df, method, col, val):
    s = df[(df["method"] == method) & (np.isclose(df[col], val))]
    t = s[s["round"] > s["round"].max() - 10]
    return t.groupby("seed")["macro_f1"].mean()


def deficit(df, col, val):
    a, b = l10(df, "proposed_hfl", col, val), l10(df, "flat_fl", col, val)
    i = sorted(set(a.index) & set(b.index))
    return float((a.loc[i] - b.loc[i]).mean())


def shard(df, col, val):
    t = df[df["round"] > df["round"].max() - 10]
    s = t[(t["method"] == "proposed_hfl") & (np.isclose(t[col], val))]
    return float(s["mean_shard_clients"].mean())


def main() -> int:
    tab = pd.read_parquet("results/paper_full_cf60/paper_sweep_rounds.parquet")
    cov = pd.read_parquet("results/paper_coverage_cf60/coverage_sweep_rounds.parquet")

    rows = []
    print("N axis (R_comm at the table's value), proposed_hfl - flat_fl:")
    print(f"{'N':>6} {'shard':>8} {'deficit':>9}")
    for n in (30, 50, 100, 200):
        sh, df_ = shard(tab, "N", n), deficit(tab, "N", n)
        rows.append(("N", n, sh, df_))
        print(f"{n:>6} {sh:>8.3f} {df_:>+9.4f}")

    print("\nR_comm axis (N fixed at 200), proposed_hfl - flat_fl:")
    print(f"{'R':>6} {'shard':>8} {'deficit':>9}")
    for r in (500, 1000, 2000, 3000, 4000, 5000):
        sh, df_ = shard(cov, "R_comm", r), deficit(cov, "R_comm", r)
        rows.append(("R", r, sh, df_))
        print(f"{r:>6} {sh:>8.3f} {df_:>+9.4f}")

    print("\nmatched on shard width — if shard width mediates, these should agree:")
    ns = [x for x in rows if x[0] == "N"]
    rs = [x for x in rows if x[0] == "R"]
    for _, n, shn, dn in ns:
        near = min(rs, key=lambda x: abs(x[2] - shn))
        print(f"  shard~{shn:.2f}:  N={n:<4} {dn:+.4f}   vs   "
              f"R={near[1]:<5} (shard {near[2]:.2f}) {near[3]:+.4f}   "
              f"gap {abs(dn - near[3]):.4f}")

    both = rows
    r = np.corrcoef([x[2] for x in both], [x[3] for x in both])[0, 1]
    print(f"\npooled correlation(shard, deficit) over all 10 cells: r = {r:+.3f}")
    rn = np.corrcoef([x[2] for x in ns], [x[3] for x in ns])[0, 1]
    rr = np.corrcoef([x[2] for x in rs], [x[3] for x in rs])[0, 1]
    print(f"  within N axis: r = {rn:+.3f}     within R axis: r = {rr:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
