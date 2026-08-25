#!/usr/bin/env python
"""Score the fusion-ownership arm against the criteria in configs/fusion_owner.yaml.

Two pre-registered contrasts, Holm-corrected within each (4 N-values each):

  C1 = B - A : proposed_hfl(fusion_owner=client) - proposed_hfl(fusion_owner=uav)
               Does moving fusion OFF the UAV tier help?
  C2 = B - C : proposed_hfl(fusion_owner=client) - flat_fl
               Does UAV-trained img_proj beat a frozen random projection?

A and C are read from results/paper_full and are NOT recomputed; that is what
keeps both pairings exact (same method name -> same seed stream, and flat_fl
ignores fusion_owner because it has no UAV tier).

Decision rules, fixed before the seeds existed:
  1. C1 > 0 and Holm-significant at >= 3 of 4 N -> UAV-owned fusion IS the mechanism
  2. C2 >= 0 (not significantly negative) at >= 3 of 4 N -> the image tier pays for itself
  3. C1 null -> UAV-owned fusion is NOT the mechanism; scope-condition framing

N=200 is pre-declared underpowered (A-C gap 0.012 vs a median MDE of ~0.034 at
n=10). A null there is reported as uninformative, never as "no effect".
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
    """mean(a - b), Wilcoxon p, n_pairs. Positive favours `a`."""
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


def contrast(name: str, left: pd.DataFrame, lmeth: str,
             right: pd.DataFrame, rmeth: str) -> dict:
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
            note = "below MDE - uninformative"
        print(f"{n:>5}  {diffs[n]:>+9.4f}  {pvals[n]:>9.4f}  "
              f"{str(sig[n]):>5}  {npairs[n]:>5}  {note}")
    return {"diffs": diffs, "sig": sig}


def main() -> int:
    full, fus = load("paper_full"), load("fusion_owner")
    if full is None:
        print("results/paper_full missing - cannot pair", file=sys.stderr)
        return 2
    if fus is None:
        print("results/fusion_owner missing - run configs/fusion_owner.yaml first",
              file=sys.stderr)
        return 2

    c1 = contrast("C1  B-A   client-fusion  minus  uav-fusion (proposed_hfl)",
                  fus, "proposed_hfl", full, "proposed_hfl")
    c2 = contrast("C2  B-C   client-fusion (proposed_hfl)  minus  flat_fl",
                  fus, "proposed_hfl", full, "flat_fl")

    # Reference: the deficit this arm is trying to explain.
    print("\n=== reference  A-C   uav-fusion (proposed_hfl) minus flat_fl ===")
    for n in NS:
        d, p, _ = paired(last10(full, "proposed_hfl", n), last10(full, "flat_fl", n))
        print(f"{n:>5}  {d:>+9.4f}  p={p:.4f}")

    n_c1_win = sum(1 for n in NS if c1["sig"][n] and c1["diffs"][n] > 0)
    n_c2_ok = sum(1 for n in NS
                  if not (c2["sig"][n] and c2["diffs"][n] < 0))

    crit1 = n_c1_win >= 3
    crit2 = n_c2_ok >= 3

    print("\n" + "=" * 62)
    print(f"criterion 1  C1 Holm-positive at >=3 of 4 N : {n_c1_win}/4  "
          f"{'PASS' if crit1 else 'FAIL'}")
    print(f"criterion 2  C2 not sig-negative at >=3 of 4 N: {n_c2_ok}/4  "
          f"{'PASS' if crit2 else 'FAIL'}")
    print("=" * 62)

    if crit1:
        print("\nVERDICT: UAV-owned fusion IS the mechanism.")
        print("This is a DISCOVERY, NOT A WIN. It licenses exactly one thing:")
        print("rerunning the FULL v5 table for EVERY baseline under")
        print("fusion_owner: client, plus a re-tune, before any comparative")
        print("claim. Reporting a win off this arm alone would be selecting the")
        print("configuration after seeing the answer.")
    else:
        print("\nVERDICT: rule 3 - UAV-owned fusion is NOT the mechanism.")
        print("The deficit lies in the image pathway itself or in the geographic")
        print("sharding. Report the hierarchy as net-negative at coherent radius")
        print("(scope-condition framing).")
    if not crit2:
        print("\nC2 significantly negative: UAV-trained img_proj is WORSE than")
        print("flat_fl's frozen random projection. The image tier does not pay")
        print("for itself at this radius regardless of who owns fusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
