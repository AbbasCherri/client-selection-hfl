#!/usr/bin/env python
"""Minimum detectable effect for every null the paper leans on.

A paper whose central claims are negative has to answer one question before any
other: could it have detected the effect it says is absent? "p > 0.05 at n=10"
is not evidence of absence unless the design could have seen an effect worth
caring about.

So for each null, report the observed paired difference, the SD of those paired
differences, and the effect the design could detect at 80% power. Then a reader
can see directly whether the null is informative ("we would have caught anything
above X, and X is small") or uninformative ("we would only have caught something
implausibly large").

MDE uses the paired-t approximation
    MDE = (t_{1-a/2, n-1} + t_{power, n-1}) * sd / sqrt(n)
which at n=10, alpha=0.05, power=0.80 is almost exactly 1.0 * sd. The reported
tests are Wilcoxon (no normality assumption); the t approximation is used only
to size the detectable effect, is standard for that purpose, and is mildly
CONSERVATIVE relative to the signed-rank test on well-behaved differences —
i.e. it does not flatter the nulls.

Usage:  python scripts/power_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.stats import wilcoxon

ALPHA, POWER = 0.05, 0.80


def mde(sd: float, n: int) -> float:
    if n < 2:
        return float("nan")
    return (tdist.ppf(1 - ALPHA / 2, n - 1) + tdist.ppf(POWER, n - 1)) * sd / np.sqrt(n)


def last10(df: pd.DataFrame, keys) -> pd.Series:
    tail = df[df["round"] > df["round"].max() - 10]
    return tail.groupby(list(keys))["macro_f1"].mean()


def compare(label: str, a: pd.Series, b: pd.Series, rows: list) -> None:
    shared = sorted(set(a.index) & set(b.index))
    if len(shared) < 3:
        return
    d = a.loc[shared].to_numpy() - b.loc[shared].to_numpy()
    n = len(d)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    sd = float(d.std(ddof=1))
    m = mde(sd, n)
    rows.append({
        "comparison": label, "n": n,
        "observed": round(float(d.mean()), 4),
        "sd_diff": round(sd, 4),
        "p": round(p, 4),
        "MDE@80%": round(m, 4),
        "observed/MDE": round(abs(float(d.mean())) / m, 2) if m > 0 else np.nan,
    })


def main() -> int:
    rows: list[dict] = []

    # --- 1. class realism: the contrast that inverted the paper's claim -----
    arms = {
        "true_placeaware": "class_realism_true",
        "true_placeblind": "class_realism_noplace",
        "pseudo_placeaware": "class_realism_pseudo",
        "none_placeblind": "class_realism_none",
    }
    got = {}
    for tag, d in arms.items():
        pq = sorted(Path("results", d).glob("*.parquet"))
        if pq:
            got[tag] = last10(pd.read_parquet(pq[0]), ["seed"])
    if {"true_placeaware", "true_placeblind"} <= got.keys():
        compare("class-realism SELECTION (true_placeblind - none_placeblind)",
                got.get("true_placeblind"), got.get("none_placeblind"), rows) \
            if "none_placeblind" in got else None
        compare("class-realism PLACEMENT (true_placeaware - true_placeblind)",
                got["true_placeaware"], got["true_placeblind"], rows)
    if {"pseudo_placeaware", "none_placeblind"} <= got.keys():
        compare("class-realism pseudo vs none",
                got["pseudo_placeaware"], got["none_placeblind"], rows)

    # --- 2. v6 / C3 arms vs mclp_place, per K ------------------------------
    base = Path("results/paper_uav_count/uav_sweep_rounds.parquet")
    if base.exists():
        b = pd.read_parquet(base)
        mclp = b[b["method"] == "mclp_place"]
        moon = b[b["method"] == "moon2022"]
        for arm, tag in (("v6_c1_reachable", "C1"), ("v6_c2_diversity", "C2"),
                         ("v6_both", "C1+C2"), ("v6_c3_disjoint", "C3")):
            pq = sorted(Path("results", arm).glob("*.parquet"))
            if not pq:
                continue
            a = pd.read_parquet(pq[0])
            for K in sorted(a["K"].unique()):
                compare(f"{tag} - mclp_place @K={K}",
                        last10(a[a["K"] == K], ["seed"]),
                        last10(mclp[mclp["K"] == K], ["seed"]), rows)
            for K in (10, 15, 20):
                compare(f"{tag} - moon2022 @K={K}",
                        last10(a[a["K"] == K], ["seed"]),
                        last10(moon[moon["K"] == K], ["seed"]), rows)

    if not rows:
        print("no results found — run from the repo root on the VM")
        return 1

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 300)
    print("=== minimum detectable effect, paired, n as shown, alpha=0.05, power=0.80 ===\n")
    print(out.to_string(index=False))

    print("\n=== how to read this ===")
    print("observed/MDE < 1 : the design could NOT have detected an effect this")
    print("                   small; the null is 'not detected', never 'absent'.")
    print("observed/MDE > 1 : an effect of the observed size WOULD have been")
    print("                   caught, so a non-significant result is informative.")
    med = out["MDE@80%"].median()
    print(f"\nmedian MDE across these comparisons: {med:.4f} macro-F1")
    print("Quote this beside every null the paper reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
