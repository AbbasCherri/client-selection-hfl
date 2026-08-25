#!/usr/bin/env python
"""Score a within-arm selector comparison. Usage: score_selectors.py <arm_dirname>

Unlike score_arm.py (which pairs an arm against results/paper_full), this scores
methods AGAINST EACH OTHER inside a single run, which is what the paper's actual
claim is about: does the proposed SELECTION rule beat the literature selectors?

Criteria, fixed in the arm's config before its first seed existed:
  1. proposed_hfl Holm-beats fedcs at >= 3 of 4 N
  2. proposed_hfl Holm-beats oort  at >= 3 of 4 N
  3. proposed_hfl NOT significantly worse than flat_fl at >= 3 of 4 N

Holm is applied within each pairwise family (4 N-values), matching how the
paper's significance tables are corrected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NS = (30, 50, 100, 200)
MDE_N10 = 0.034
REF = "proposed_hfl"


def load(stem: str):
    pq = sorted(Path("results", stem).glob("*.parquet"))
    return pd.read_parquet(pq[0]) if pq else None


def last10(df, method, n):
    sub = df[(df["method"] == method) & (df["N"] == n)]
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


def family(df, opponent):
    """proposed_hfl minus opponent, Holm-corrected across the 4 N-values."""
    diffs, pvals, npairs = {}, {}, {}
    for n in NS:
        d, p, k = paired(last10(df, REF, n), last10(df, opponent, n))
        diffs[n], pvals[n], npairs[n] = d, p, k
    sig = holm(pvals)
    print(f"\n=== {REF} minus {opponent} ===")
    print(f"{'N':>5}  {'diff':>9}  {'p':>9}  {'Holm':>5}  {'pairs':>5}  note")
    for n in NS:
        note = ""
        if not np.isnan(diffs[n]) and abs(diffs[n]) < MDE_N10 and not sig[n]:
            note = "below MDE - uninformative on small effects"
        print(f"{n:>5}  {diffs[n]:>+9.4f}  {pvals[n]:>9.4f}  "
              f"{str(sig[n]):>5}  {npairs[n]:>5}  {note}")
    return diffs, sig


def main() -> int:
    arm = sys.argv[1] if len(sys.argv) > 1 else "selectors_k10"
    df = load(arm)
    if df is None:
        print(f"results/{arm} missing - run its config first", file=sys.stderr)
        return 2

    present = set(df["method"].unique())
    for m in (REF, "fedcs", "oort", "flat_fl"):
        if m not in present:
            print(f"method {m} absent from results/{arm} - cannot score",
                  file=sys.stderr)
            return 2

    d_fedcs, s_fedcs = family(df, "fedcs")
    d_oort, s_oort = family(df, "oort")
    d_flat, s_flat = family(df, "flat_fl")

    if "mean_shard_clients" in df.columns:
        print("\n=== mediator: mean_shard_clients (the lever this arm moves) ===")
        for n in NS:
            v = df[(df["method"] == REF) & (df["N"] == n)]["mean_shard_clients"].mean()
            print(f"  N={n:<4} {v:.2f}")

    n1 = sum(1 for n in NS if s_fedcs[n] and d_fedcs[n] > 0)
    n2 = sum(1 for n in NS if s_oort[n] and d_oort[n] > 0)
    n3 = sum(1 for n in NS if not (s_flat[n] and d_flat[n] < 0))
    c1, c2, c3 = n1 >= 3, n2 >= 3, n3 >= 3

    print("\n" + "=" * 68)
    print(f"criterion 1  beats fedcs   at >=3 of 4 N : {n1}/4  {'PASS' if c1 else 'FAIL'}")
    print(f"criterion 2  beats oort    at >=3 of 4 N : {n2}/4  {'PASS' if c2 else 'FAIL'}")
    print(f"criterion 3  not worse than flat_fl      : {n3}/4  {'PASS' if c3 else 'FAIL'}")
    print("=" * 68)

    if c1 and c2 and c3:
        print(f"\nVERDICT: {arm} PASSES its pre-registered criteria.")
        print("This is a DISCOVERY, NOT A WIN. PREREGISTRATION.md §2 requires two")
        print("further gates: replication on fresh seeds 10-24, and Holm survival")
        print("in the final reported family. §3 then requires the full table")
        print("regenerated at this operating point and every method re-tuned here,")
        print("since current recipes were searched at R_comm=20km and K=20.")
        print("Report the coverage cost of this operating point alongside the win.")
    else:
        print(f"\nVERDICT: {arm} FAILS its pre-registered criteria.")
        print("Per this arm's config, failure of criteria 1-2 means the selection")
        print("rule does not separate from the literature even under favourable")
        print("pooling. Do NOT try further K values: the existing fleet sweep")
        print("already spans K=5..30, and searching operating points until one")
        print("passes is the optional stopping PREREGISTRATION.md forbids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
