"""Apply REPORTS/preregistration_v6_c3.md's falsification conditions to the screen.

The pre-registration names two hypotheses and says, in advance, what would kill
each. This script is the arithmetic of that, kept separate from the runner so the
verdict cannot drift into "whatever the numbers turned out to support".

  H-B (fitness weights)  rejected if `mclp_place` at w=(1,0,0) closes LESS THAN
                         ONE THIRD of the coverage gap to `moon2022` at BOTH K.
  H-A (dispersion)       rejected if `moon2022`'s dispersion statistics sit
                         within the spread of the fitness optimisers' at BOTH K.

H-A's "within the spread" is operationalised here, and the operationalisation is
printed with the result: `moon2022` must differ from BOTH `mclp_place` variants
on at least one of `uav_pairwise_sep_m` / `unique_cover_frac` /
`cover_multiplicity_mean`, in a consistent direction, by a paired Wilcoxon
p < 0.05 across the 10 seeds, at BOTH K. Ten seeds is thin, so a near-miss is
reported as a near-miss rather than rounded either way.

If both are rejected, C3 is not run and the mechanism is reported unidentified.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

pd.set_option("display.width", 220)

GEOM = ["coverage_pct", "cover_multiplicity_mean", "unique_cover_frac",
        "uav_pairwise_sep_m"]


def load(d: str, tag: str) -> pd.DataFrame | None:
    pq = sorted(Path("results", d).glob("*.parquet"))
    if not pq:
        print(f"!! missing results/{d} — run scripts/run_c3_screen.sh first")
        return None
    df = pd.read_parquet(pq[0])
    # One-round screen: round 1 is the only round, and the only one whose
    # geometry is meaningful. Guard rather than assume.
    df = df[df["round"] == df["round"].min()]
    df["arm"] = tag
    return df


shipped = load("c3_screen", "shipped")
w100 = load("c3_screen_w100", "w100")
if shipped is None or w100 is None:
    sys.exit(1)

mclp = shipped[shipped["method"] == "mclp_place"].copy()
mclp["arm"] = "mclp_shipped"
moon = shipped[shipped["method"] == "moon2022"].copy()
moon["arm"] = "moon2022"
w100 = w100[w100["method"] == "mclp_place"].copy()
w100["arm"] = "mclp_w100"
allrows = pd.concat([mclp, moon, w100], ignore_index=True)

print("=== screen means by arm and K ===")
for col in GEOM:
    if col not in allrows.columns:
        print(f"  (missing column {col})")
        continue
    print(f"--- {col} ---")
    print(allrows.pivot_table(index="K", columns="arm", values=col).round(3).to_string())
print()

Ks = sorted(allrows["K"].unique())


def by_seed(arm: str, K: int, col: str) -> pd.Series:
    s = allrows[(allrows["arm"] == arm) & (allrows["K"] == K)]
    return s.set_index("seed")[col].sort_index()


# ---------------- H-B: the fitness weights ----------------
print("=== H-B — does removing the energy/imbalance terms close the coverage gap? ===")
hb_closed = {}
for K in Ks:
    m = by_seed("mclp_shipped", K, "coverage_pct").mean()
    w = by_seed("mclp_w100", K, "coverage_pct").mean()
    mo = by_seed("moon2022", K, "coverage_pct").mean()
    gap = mo - m
    frac = (w - m) / gap if abs(gap) > 1e-9 else np.nan
    hb_closed[K] = frac
    print(f"  K={K:<3} mclp {m:6.2f}  ->  w100 {w:6.2f}   moon {mo:6.2f}   "
          f"gap {gap:+.2f} pp, closed {frac:.0%}")
hb_rejected = all((not np.isfinite(f)) or f < 1 / 3 for f in hb_closed.values())
print(f"  => H-B {'REJECTED' if hb_rejected else 'SURVIVES'} "
      f"(rejected only if < 1/3 closed at BOTH K)")
print()

# ---------------- H-A: dispersion ----------------
print("=== H-A — is moon2022's geometry outside the fitness optimisers' spread? ===")
ha_hits = []
for col in ("uav_pairwise_sep_m", "unique_cover_frac", "cover_multiplicity_mean"):
    if col not in allrows.columns:
        continue
    per_K = {}
    for K in Ks:
        mo = by_seed("moon2022", K, col)
        res = []
        for other in ("mclp_shipped", "mclp_w100"):
            ot = by_seed(other, K, col)
            j = pd.concat([mo.rename("a"), ot.rename("b")], axis=1).dropna()
            if len(j) < 5 or np.allclose(j["a"], j["b"]):
                res.append((np.nan, 0.0))
                continue
            p = wilcoxon(j["a"], j["b"]).pvalue
            res.append((p, float(j["a"].mean() - j["b"].mean())))
        sig = all(np.isfinite(p) and p < 0.05 for p, _ in res)
        same_dir = len({np.sign(d) for _, d in res}) == 1 and res[0][1] != 0
        per_K[K] = sig and same_dir
        print(f"  {col:<24} K={K:<3} " + "  ".join(
            f"vs {o}: d={d:+.4g} p={p:.4g}" for (p, d), o in zip(res, ("shipped", "w100"))))
    if all(per_K.values()):
        ha_hits.append(col)
ha_rejected = not ha_hits
print(f"  => H-A {'REJECTED' if ha_rejected else 'SURVIVES on ' + ', '.join(ha_hits)} "
      f"(needs one statistic separating at BOTH K vs BOTH variants)")
print()

print("=== VERDICT ===")
if hb_rejected and ha_rejected:
    print("Both hypotheses rejected by the screen. Per the pre-registration, C3 is")
    print("NOT run and the mechanism behind moon2022's advantage is reported as")
    print("unidentified. That is a legitimate outcome — write it up.")
else:
    surviving = []
    if not hb_rejected:
        surviving.append("H-B (fitness weights)")
    if not ha_rejected:
        surviving.append("H-A (dispersion)")
    print("Surviving: " + "; ".join(surviving))
    print("Run the corresponding C3 arm on the full fleet grid and judge it")
    print("against the four criteria in REPORTS/preregistration_v6_c3.md §4.")
