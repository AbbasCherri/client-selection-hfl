#!/usr/bin/env python
"""Everything the manuscript quotes from the corrected main grid, from one script.

Reads results/paper_full_cf60 (5 arms) and results/paper_full_cf60b (8 arms) as
one grid, after asserting the two resolved configurations differ only in name,
results_dir and methods. Prints three sections, written together so the
numbers cannot drift from one another:

  1. MAIN TABLE   mean macro-F1 per arm and N, the system's rank, and
                  system-minus-comparator differences with Holm within each
                  comparator's family of four client counts.
  2. MDE          per-cell minimum detectable effect for every system-vs-comparator
                  comparison, same paired-t approximation as power_analysis.py,
                  plus the median and the observed/MDE ranges the paper quotes.
  3. HORIZON      Spearman rank agreement between each 10-round window and rounds
                  91-100, averaged over N, and the system's rank per window.

Replaces the 2026-09-05 horizon table, which was computed ad hoc on the
superseded 20-round-recipe grid and never committed.

Usage:  python scripts/main_grid_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, wilcoxon
from scipy.stats import t as tdist

TREES = ("paper_full_cf60", "paper_full_cf60b")
SYSTEM = "proposed_hfl"
ORACLE = "centralized"
ALPHA, POWER = 0.05, 0.80
REDUCE = 10


def mde(sd: float, n: int) -> float:
    if n < 2:
        return float("nan")
    return (tdist.ppf(1 - ALPHA / 2, n - 1) + tdist.ppf(POWER, n - 1)) * sd / np.sqrt(n)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        prev = max(prev, p * (m - i))
        out[k] = prev < ALPHA
    return out


def check_trees() -> None:
    sigs = []
    for t in TREES:
        p = Path("results", t, "config.paper_sweep.resolved.yaml")
        if not p.exists():
            sys.exit(f"missing {p}: has {t} been run?")
        c = yaml.safe_load(p.read_text())
        for k in ("name", "results_dir", "methods"):
            c.pop(k, None)
        sigs.append(c)
    if sigs[0] != sigs[1]:
        diff = sorted(k for k in set(sigs[0]) | set(sigs[1])
                      if sigs[0].get(k) != sigs[1].get(k))
        sys.exit(f"trees are NOT concatenable; resolved configs differ in {diff}")


def load() -> pd.DataFrame:
    check_trees()
    frames = [pd.read_parquet(Path("results", t, "paper_sweep_rounds.parquet"))
              for t in TREES]
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(["N", "seed", "method", "round"]).sum()
    if dup:
        sys.exit(f"{dup} duplicate (N, seed, method, round) rows across trees")
    return df


def per_seed(df: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    w = df[(df["round"] >= lo) & (df["round"] <= hi)]
    return w.groupby(["N", "method", "seed"])["macro_f1"].mean()


def main() -> int:
    df = load()
    r0, r1 = int(df["round"].min()), int(df["round"].max())
    Ns = sorted(df["N"].unique())
    methods = sorted(df["method"].unique())
    comparators = [m for m in methods if m != SYSTEM]
    final = per_seed(df, r1 - REDUCE + 1, r1)

    print(f"# main grid: {TREES} concatenated; {len(methods)} arms; N={Ns}; "
          f"rounds {r0}-{r1}; seeds {sorted(df['seed'].unique())}")
    print(f"# reduction: mean macro-F1 over last {REDUCE} rounds per seed")

    # ---------------------------------------------------------------- 1. table
    print("\n" + "=" * 72 + "\n1. MAIN TABLE\n" + "=" * 72)
    means = final.groupby(["N", "method"]).mean().unstack("N")
    order = means[Ns[-1]].sort_values(ascending=False).index
    print(f"{'arm':>20} " + " ".join(f"{'N='+str(n):>9}" for n in Ns))
    for m in order:
        print(f"{m:>20} " + " ".join(f"{means.loc[m, n]:>9.4f}" for n in Ns))
    print()
    for n in Ns:
        col = means[n].sort_values(ascending=False)
        fed = col.drop(ORACLE, errors="ignore")
        print(f"  N={n:<4} system rank {list(col.index).index(SYSTEM)+1}/{len(col)} "
              f"of all arms, {list(fed.index).index(SYSTEM)+1}/{len(fed)} "
              f"excluding the centralized oracle")

    rows = []
    print(f"\nsystem minus comparator (* = Holm-significant within its family of "
          f"{len(Ns)} client counts)")
    print(f"{'comparator':>20} " + " ".join(f"{'N='+str(n):>10}" for n in Ns)
          + "   Holm wins/losses")
    for c in comparators:
        diffs, pv, sds, ns = {}, {}, {}, {}
        for n in Ns:
            a = final.loc[(n, SYSTEM)]
            b = final.loc[(n, c)]
            idx = sorted(set(a.index) & set(b.index))
            d = a.loc[idx].to_numpy() - b.loc[idx].to_numpy()
            diffs[n] = float(d.mean())
            pv[n] = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
            sds[n], ns[n] = float(d.std(ddof=1)), len(d)
        sig = holm(pv)
        cells = " ".join(f"{diffs[n]:>+9.4f}{'*' if sig[n] else ' '}" for n in Ns)
        w = sum(sig[n] and diffs[n] > 0 for n in Ns)
        l = sum(sig[n] and diffs[n] < 0 for n in Ns)
        print(f"{c:>20} {cells}   {w}/{l}")
        for n in Ns:
            rows.append({"comparator": c, "N": n, "diff": diffs[n], "p": pv[n],
                         "holm": sig[n], "sd": sds[n], "n": ns[n],
                         "mde": mde(sds[n], ns[n])})
    res = pd.DataFrame(rows)
    res["obs_over_mde"] = res["diff"].abs() / res["mde"]

    wins = res[res["holm"] & (res["diff"] > 0)]
    print(f"\n  system Holm-WINS anywhere: {len(wins)}"
          + ("" if wins.empty else "  -> " + ", ".join(
              f"{r.comparator}@N={r.N}" for r in wins.itertuples())))
    for c in comparators:
        s = res[res["comparator"] == c]
        if (s["holm"] & (s["diff"] < 0)).all():
            print(f"  system Holm-LOSES to {c} at all {len(Ns)} client counts")
    floor = 2 / 2 ** int(res["n"].min())
    print(f"\n  Holm family = {len(Ns)} client counts per comparator; first "
          f"threshold {ALPHA/len(Ns):.3e}; two-sided Wilcoxon floor at "
          f"n={int(res['n'].min())} is {floor:.3e} -> "
          f"{'reachable' if floor < ALPHA/len(Ns) else 'UNREACHABLE'}")

    # --------------------------------------------------------------------- 2. MDE
    print("\n" + "=" * 72 + "\n2. MDE (paired-t approximation, 80% power)\n" + "=" * 72)
    print(f"  comparisons: {len(res)} ({len(comparators)} comparators x {len(Ns)} N)")
    print(f"  median MDE over all comparisons: {res['mde'].median():.4f}")
    print(f"  median MDE excluding the oracle: "
          f"{res[res['comparator'] != ORACLE]['mde'].median():.4f}")
    print(f"  MDE range: {res['mde'].min():.4f} - {res['mde'].max():.4f}")
    for grp, members in (("flat_fl and hfl_no_selection", ["flat_fl", "hfl_no_selection"]),
                         ("all Holm-significant losses", None)):
        s = res[res["comparator"].isin(members)] if members else \
            res[res["holm"] & (res["diff"] < 0)]
        if s.empty:
            continue
        print(f"  losses to {grp}: diff {s['diff'].min():+.4f} to {s['diff'].max():+.4f}; "
              f"obs/MDE {s['obs_over_mde'].min():.2f} - {s['obs_over_mde'].max():.2f}")
    below = res[res["holm"] & (res["obs_over_mde"] < 1)]
    if not below.empty:
        print("  Holm-significant but below MDE (consistent sign, small magnitude):")
        for r in below.itertuples():
            print(f"    {r.comparator}@N={r.N}: {r.diff:+.4f}, MDE {r.mde:.4f}, "
                  f"obs/MDE {r.obs_over_mde:.2f}")
    out = Path("results/main_grid_mde.csv")
    res.round(6).to_csv(out, index=False)
    print(f"  per-cell table -> {out}")

    # ----------------------------------------------------------------- 3. horizon
    print("\n" + "=" * 72 + "\n3. HORIZON\n" + "=" * 72)
    windows = [(lo, lo + 9) for lo in range(r0, r1 - 9, 10)]
    ref = final.groupby(["N", "method"]).mean()
    for label, drop in (("excluding the centralized oracle (primary)", {ORACLE}),
                        ("all arms, oracle included", set())):
        arms = [m for m in methods if m not in drop]
        print(f"\n  {label}: {len(arms)} arms; Spearman rho vs rounds "
              f"{r1-REDUCE+1}-{r1}, averaged over N")
        print(f"  {'window':>10} {'rho':>7}   system rank per N " + str(Ns))
        for lo, hi in windows:
            w = per_seed(df, lo, hi).groupby(["N", "method"]).mean()
            rhos, ranks = [], []
            for n in Ns:
                x = [w.loc[(n, m)] for m in arms]
                y = [ref.loc[(n, m)] for m in arms]
                rhos.append(float(spearmanr(x, y)[0]))
                ser = pd.Series(x, index=arms).sort_values(ascending=False)
                ranks.append(list(ser.index).index(SYSTEM) + 1)
            print(f"  {lo:>4}-{hi:<5} {np.mean(rhos):>+7.2f}   "
                  + " ".join(f"{r:>2}" for r in ranks) + f"  (of {len(arms)})")
        k = len(arms)
        print(f"  granularity: one adjacent rank swap moves rho by "
              f"{12/(k*(k**2-1)):.3f} at k={k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
