"""Score the decisive table: proposed_hfl under symmetric 60-round recipes.

Reduction = mean macro-F1 over the last 10 rounds per seed. Paired Wilcoxon
across evaluation seeds 0-9. Holm correction within each comparator family
(the 4 values of N), matching every prior scorer in this project.

Evaluates the three pre-registered criteria verbatim:
  1. proposed_hfl Holm-beats fedcs at >= 3 of 4 N
  2. proposed_hfl Holm-beats oort  at >= 3 of 4 N
  3. proposed_hfl NOT significantly worse than flat_fl at >= 3 of 4 N

and reports hfl_no_selection, which isolates what the selection rule itself
contributes and has beaten the proposed rule in every measurement so far.

Prints the previous table (regime B recipes) beside it so the effect of the
retune is visible rather than asserted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NS = (30, 50, 100, 200)
MDE_N10 = 0.034
COMPARATORS = ("fedcs", "oort", "flat_fl", "hfl_no_selection")


def load(stem):
    p = Path("results", stem, "paper_sweep_rounds.parquet")
    if p.exists():
        return pd.read_parquet(p)
    q = sorted(Path("results", stem).glob("*.parquet"))
    return pd.read_parquet(q[0]) if q else None


def last10(df, method, n):
    sub = df[(df["method"] == method) & (df["N"] == n)]
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


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        prev = max(prev, p * (m - i))
        out[k] = prev < 0.05
    return out


def family(df, comp):
    diffs, pvals, npairs = {}, {}, {}
    for n in NS:
        d, p, k = paired(last10(df, "proposed_hfl", n), last10(df, comp, n))
        diffs[n], pvals[n], npairs[n] = d, p, k
    return diffs, pvals, holm(pvals), npairs


def main() -> int:
    new = load("paper_full_cf60")
    if new is None:
        print("no results at results/paper_full_cf60")
        return 1

    # the table this replaces: r2 covers the 4 carrying methods on seeds 0-9,
    # paper_full_cf covers hfl_no_selection on the same seeds
    old_r2 = load("r2_client_full")
    old_cf = load("paper_full_cf")

    print("# DECISIVE TABLE — results/paper_full_cf60")
    print("# symmetric 60-round recipes (configs/tuned_weights_cf60.yaml),")
    print("# fusion_owner=client_full, evaluation seeds 0-9")
    print(f"# {len(new)} rows, methods {sorted(new['method'].unique())}")

    results = {}
    for comp in COMPARATORS:
        if comp not in set(new["method"]):
            continue
        diffs, pvals, sig, npairs = family(new, comp)
        results[comp] = (diffs, sig)
        print(f"\n=== proposed_hfl - {comp} ===")
        print(f"{'N':>5}  {'diff':>9}  {'p':>9}  {'Holm':>6}  {'n':>3}  note")
        for n in NS:
            note = ""
            if not np.isnan(diffs[n]) and abs(diffs[n]) < MDE_N10 and not sig[n]:
                note = "below MDE - uninformative"
            print(f"{n:>5}  {diffs[n]:>+9.4f}  {pvals[n]:>9.4f}  "
                  f"{str(sig[n]):>6}  {npairs[n]:>3}  {note}")
        prev = old_r2 if (old_r2 is not None and comp in set(old_r2["method"])) else old_cf
        if prev is not None and comp in set(prev["method"]):
            pd_, pp_, ps_, _ = family(prev, comp)
            cells = "  ".join(format(pd_[n], "+.4f") + ("*" if ps_[n] else " ")
                              for n in NS)
            print(f"  previous table (20-round uav recipes): {cells}")

    print("\n" + "=" * 62)
    print("PRE-REGISTERED CRITERIA")
    print("=" * 62)
    verdicts = {}
    if "fedcs" in results:
        d, s = results["fedcs"]
        w = sum(1 for n in NS if s[n] and d[n] > 0)
        verdicts[1] = w >= 3
        print(f"  1. Holm-beats fedcs at >=3 of 4 N        : {w}/4  "
              f"{'PASS' if w >= 3 else 'FAIL'}")
    if "oort" in results:
        d, s = results["oort"]
        w = sum(1 for n in NS if s[n] and d[n] > 0)
        verdicts[2] = w >= 3
        print(f"  2. Holm-beats oort at >=3 of 4 N         : {w}/4  "
              f"{'PASS' if w >= 3 else 'FAIL'}")
    if "flat_fl" in results:
        d, s = results["flat_fl"]
        ok = sum(1 for n in NS if not (s[n] and d[n] < 0))
        verdicts[3] = ok >= 3
        print(f"  3. Not sig. worse than flat_fl at >=3 of 4: {ok}/4  "
              f"{'PASS' if ok >= 3 else 'FAIL'}")
    allpass = all(verdicts.values()) if verdicts else False
    print(f"\n  COMPOSITE: {'PASS' if allpass else 'FAIL'} "
          f"({sum(verdicts.values())}/{len(verdicts)} criteria met)")

    if "hfl_no_selection" in results:
        d, s = results["hfl_no_selection"]
        losses = sum(1 for n in NS if s[n] and d[n] < 0)
        print(f"\n  hfl_no_selection (not a criterion): proposed_hfl is "
              f"Holm-WORSE at {losses}/4 N")
        print("  This is the ablation that isolates the selection rule: same "
              "hierarchy,\n  no selector. Losing to it means the rule does not "
              "pay for itself.")

    print("\n=== ranking by mean macro-F1 (last 10 rounds, pooled over seeds) ===")
    tail = new[new["round"] > new["round"].max() - 10]
    for n in NS:
        r = (tail[tail["N"] == n].groupby("method")["macro_f1"]
             .mean().sort_values(ascending=False))
        pos = list(r.index).index("proposed_hfl") + 1 if "proposed_hfl" in r.index else 0
        print(f"  N={n:<4} (proposed_hfl ranks {pos}/{len(r)}): "
              + "  ".join(f"{m}={v:.4f}" for m, v in r.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
