#!/usr/bin/env python
"""Generic arm scorer. Usage: score_arm.py <arm_results_dirname>

Applies the criteria structure fixed in PREREGISTRATION.md §2 and in the arm's
own config, which was committed before that arm's first seed existed.

  C1 = B - A : arm's proposed_hfl  minus  paper_full's proposed_hfl
  C2 = B - C : arm's proposed_hfl  minus  paper_full's flat_fl

A and C are read from results/paper_full and are NOT recomputed; that is what
keeps both pairings exact (same method name -> same seed stream, and flat_fl is
registered as (None, "all", ...) so it is unaffected by placement, capacity or
fusion ownership).

Criteria, identical in structure for every arm:
  1. C1 > 0 and Holm-significant at >= 3 of 4 N
  2. C2 >= 0 (not significantly negative) at >= 3 of 4 N

Verdict text is deliberately ARM-NEUTRAL. An earlier version carried H1's
wording into every arm and printed "UAV-owned fusion is NOT the mechanism"
while scoring a capacity arm. What a failure MEANS is the arm config's job to
state, not this script's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NS = (30, 50, 100, 200)
MDE_N10 = 0.034  # median minimum detectable effect at n=10 on this harness


def load(stem: str) -> pd.DataFrame | None:
    pq = sorted(Path("results", stem).glob("*.parquet"))
    return pd.read_parquet(pq[0]) if pq else None


def last10(df: pd.DataFrame, method: str, n: int) -> pd.Series:
    sub = df[(df["method"] == method) & (df["N"] == n)]
    if sub.empty:
        return pd.Series(dtype=float)
    tail = sub[sub["round"] > sub["round"].max() - 10]
    return tail.groupby("seed")["macro_f1"].mean()


def paired(a: pd.Series, b: pd.Series):
    idx = sorted(set(a.index) & set(b.index))
    if len(idx) < 3:
        return float("nan"), 1.0, len(idx)
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    if np.allclose(x, y):
        return 0.0, 1.0, len(idx)
    return float((x - y).mean()), float(wilcoxon(x, y).pvalue), len(idx)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        prev = max(prev, p * (m - i))
        out[k] = prev < 0.05
    return out


def contrast(name, left, lmeth, right, rmeth) -> dict:
    diffs, pvals, npairs = {}, {}, {}
    for n in NS:
        d, p, k = paired(last10(left, lmeth, n), last10(right, rmeth, n))
        diffs[n], pvals[n], npairs[n] = d, p, k
    sig = holm(pvals)
    print(f"\n=== {name} ===")
    print(f"{'N':>5}  {'diff':>9}  {'p':>9}  {'Holm':>5}  {'pairs':>5}  note")
    for n in NS:
        note = ""
        if not np.isnan(diffs[n]) and abs(diffs[n]) < MDE_N10 and not sig[n]:
            note = "below MDE - uninformative on small effects"
        print(f"{n:>5}  {diffs[n]:>+9.4f}  {pvals[n]:>9.4f}  "
              f"{str(sig[n]):>5}  {npairs[n]:>5}  {note}")
    return {"diffs": diffs, "sig": sig}


def main() -> int:
    arm = sys.argv[1] if len(sys.argv) > 1 else "fusion_owner"
    full, armdf = load("paper_full"), load(arm)
    if full is None:
        print("results/paper_full missing - cannot pair", file=sys.stderr)
        return 2
    if armdf is None:
        print(f"results/{arm} missing - run its config first", file=sys.stderr)
        return 2

    c1 = contrast(f"C1  B-A   {arm} minus paper_full (proposed_hfl)",
                  armdf, "proposed_hfl", full, "proposed_hfl")
    c2 = contrast(f"C2  B-C   {arm} (proposed_hfl) minus flat_fl",
                  armdf, "proposed_hfl", full, "flat_fl")

    print("\n=== reference  A-C   paper_full proposed_hfl minus flat_fl "
          "(the deficit this arm targets) ===")
    ref = {}
    for n in NS:
        d, p, _ = paired(last10(full, "proposed_hfl", n), last10(full, "flat_fl", n))
        ref[n] = d
        print(f"{n:>5}  {d:>+9.4f}  p={p:.4f}")

    n_c1_win = sum(1 for n in NS if c1["sig"][n] and c1["diffs"][n] > 0)
    n_c2_ok = sum(1 for n in NS if not (c2["sig"][n] and c2["diffs"][n] < 0))
    crit1, crit2 = n_c1_win >= 3, n_c2_ok >= 3

    print("\n" + "=" * 66)
    print(f"criterion 1  C1 Holm-positive at >=3 of 4 N : {n_c1_win}/4  "
          f"{'PASS' if crit1 else 'FAIL'}")
    print(f"criterion 2  C2 not sig-negative at >=3 of 4 N: {n_c2_ok}/4  "
          f"{'PASS' if crit2 else 'FAIL'}")
    print("=" * 66)

    if crit1 and crit2:
        print(f"\nVERDICT: {arm} PASSES its pre-registered criteria.")
        print("This is a DISCOVERY, NOT A WIN. PREREGISTRATION.md §2 requires two")
        print("more gates before any comparative claim: replication on fresh seeds")
        print("(10-24, never used for discovery), and Holm survival in the final")
        print("reported family. §3 P2 then requires the full table rerun for every")
        print("baseline under this configuration, plus a re-tune.")
    else:
        print(f"\nVERDICT: {arm} FAILS its pre-registered criteria (rule 3).")
        print("What that failure means for the mechanism is stated in this arm's")
        print("own config, fixed before the run. It is NOT a licence to add a new")
        print("arm: see PREREGISTRATION.md §6.")

    # Whether the arm closed any of the gap it targeted — reported for every arm,
    # pass or fail, because "significant" and "enough to matter" are different.
    print("\nGap closed vs the deficit this arm targets (C2 - reference):")
    for n in NS:
        if np.isnan(c2["diffs"][n]) or np.isnan(ref[n]) or abs(ref[n]) < 1e-9:
            continue
        closed = (c2["diffs"][n] - ref[n]) / abs(ref[n]) * 100.0
        print(f"  N={n:<4} arm {c2['diffs'][n]:+.4f} vs deficit {ref[n]:+.4f}"
              f"  -> {closed:+.0f}% of the gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
