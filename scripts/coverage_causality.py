#!/usr/bin/env python
"""Is the coverage-accuracy association causal? Observational slope vs intervention.

This produces the paper's central quantitative result, so it is a committed,
provenance-linked artifact rather than an analysis someone ran once.

The argument in three steps:

  1. OBSERVATIONAL. Across the placement methods at a fixed fleet size,
     `coverage_pct` is by far the best predictor of macro-F1. Fit that slope per
     K over the non-`flat_fl` methods of the fleet sweep, EXCLUDING the
     intervention arms, so the prediction is made by data that knows nothing
     about them.

  2. INTERVENTIONAL. The v6 C1 arm changes only the placement objective
     (`coverage_mode: assigned -> reachable`) and moves coverage by a measured
     amount at each K. The slope from step 1 turns that into a predicted
     accuracy gain.

  3. COMPARE. Predicted against realised. If the association were causal the
     ratio would be near 1 at every K.

Also reported: where `moon2022` sits relative to the same per-K line. A method
whose advantage is explained by its coverage sits ON the line; a positive
residual is advantage the coverage cannot account for.

Writes a CSV so the paper's numbers come from a file rather than a screenshot.

Usage:  python scripts/coverage_causality.py [--out results/coverage_causality]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REDUCE_LAST = 10
BASE = Path("results/paper_uav_count/uav_sweep_rounds.parquet")
C1 = Path("results/v6_c1_reachable")


def last10(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["round"] > df["round"].max() - REDUCE_LAST]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/coverage_causality")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not BASE.exists():
        print(f"!! {BASE} missing")
        return 1
    v5 = last10(pd.read_parquet(BASE))
    pq = sorted(C1.glob("*.parquet"))
    if not pq:
        print(f"!! {C1} missing — run the v6 C1 arm first")
        return 1
    c1 = last10(pd.read_parquet(pq[0]))

    # ---- step 1: the mediator ranking, so the choice of coverage is justified
    MED = ["coverage_pct", "n_active_uavs", "shard_class_entropy",
           "mean_shard_clients", "jain_fairness", "n_unique_selected",
           "shard_minority_share"]
    MED = [m for m in MED if m in v5.columns]
    cell = v5.groupby(["method", "K"], as_index=False)[["macro_f1", *MED]].mean()
    hier = cell[cell["method"] != "flat_fl"]
    med_rows = []
    for m in MED:
        within = [np.corrcoef(g[m], g["macro_f1"])[0, 1]
                  for _, g in hier.groupby("K") if g[m].std() > 0]
        med_rows.append({
            "mediator": m,
            "r_pooled": round(float(np.corrcoef(hier[m], hier["macro_f1"])[0, 1]), 3),
            "r_within_K_mean": round(float(np.mean(within)), 3),
            "r_within_K_min": round(float(np.min(within)), 3),
            "r_within_K_max": round(float(np.max(within)), 3),
        })
    med = pd.DataFrame(med_rows).sort_values("r_within_K_mean", key=abs, ascending=False)
    print(f"=== mediator ranking across {len(hier)} hierarchical cells ===")
    print(med.to_string(index=False))
    med.to_csv(out / "mediators.csv", index=False)

    # ---- steps 2 and 3: slope, prediction, realisation
    rows = []
    for K in sorted(c1["K"].unique()):
        cells = (v5[(v5["K"] == K) & (v5["method"] != "flat_fl")]
                 .groupby("method", as_index=False)[["macro_f1", "coverage_pct"]].mean())
        slope, intercept = np.polyfit(cells["coverage_pct"], cells["macro_f1"], 1)
        r = float(np.corrcoef(cells["coverage_pct"], cells["macro_f1"])[0, 1])
        base = cells[cells["method"] == "mclp_place"].iloc[0]
        arm = c1[c1["K"] == K]
        d_cov = float(arm["coverage_pct"].mean() - base["coverage_pct"])
        d_f1 = float(arm["macro_f1"].mean() - base["macro_f1"])
        pred = slope * d_cov
        moon = cells[cells["method"] == "moon2022"].iloc[0]
        resid = float(moon["macro_f1"] - (slope * moon["coverage_pct"] + intercept))
        resids = cells["macro_f1"] - (slope * cells["coverage_pct"] + intercept)
        rows.append({
            "K": K, "n_methods": len(cells), "r": round(r, 3),
            "slope_per_pp": round(float(slope), 5),
            "C1_d_coverage_pp": round(d_cov, 2),
            "predicted_d_f1": round(float(pred), 4),
            "realised_d_f1": round(d_f1, 4),
            "realised_frac": round(d_f1 / pred, 2) if abs(pred) > 1e-9 else np.nan,
            "moon2022_residual": round(resid, 4),
            "moon2022_resid_rank": int((resids > resid).sum()) + 1,
        })

    res = pd.DataFrame(rows)
    print("\n=== observational slope vs C1's interventional result ===")
    print(res.to_string(index=False))
    res.to_csv(out / "causality.csv", index=False)

    tp, tr = res["predicted_d_f1"].sum(), res["realised_d_f1"].sum()
    print(f"\nsummed over K: predicted {tp:+.4f}, realised {tr:+.4f} "
          f"= {tr / tp:.0%} of prediction")
    print("\nThe aggregate is the quantity to quote. Per-K cells are individually")
    print("underpowered (see scripts/power_analysis.py) and the realised fraction")
    print("is not even consistent in sign across them — which is itself the point:")
    print("an association of this strength should not behave this way under a")
    print("direct intervention on the mediator.")
    (out / "summary.txt").write_text(
        f"predicted {tp:+.6f}\nrealised {tr:+.6f}\nfraction {tr / tp:.4f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
