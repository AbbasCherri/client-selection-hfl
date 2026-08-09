#!/usr/bin/env python
"""Both v6 switches OFF must reproduce v5 exactly.

The C1/C2 edits touched `greedy_assignment`, `AssignmentResult` and BOTH of
`Fitness`'s scoring paths (`components` and the vectorised `batch`). This
repository has a documented history of refactors that were believed
bit-identical and had to be verified rather than assumed.

If `results/v6_control` does not match `results/paper_uav_count`'s `mclp_place`
cells at the same K and seeds, the v6 comparison is measuring the refactor and
not the method, and nothing downstream of it can be interpreted.

Exit 0 if the control reproduces v5, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOL = 1e-9  # nothing in the edit path consumes RNG or reorders a float sum


def _last10(df: pd.DataFrame) -> pd.Series:
    """Mean of the last 10 rounds per seed — the significance pipeline's reduction."""
    tail = df[df["round"] > df["round"].max() - 10]
    return tail.groupby("seed")["macro_f1"].mean()


def main() -> int:
    ctl_p = Path("results/v6_control/uav_sweep_rounds.parquet")
    ref_p = Path("results/paper_uav_count/uav_sweep_rounds.parquet")
    for p in (ctl_p, ref_p):
        if not p.exists():
            print(f"[v6-control] missing {p} — cannot verify the control")
            return 1

    ctl = pd.read_parquet(ctl_p)
    ref = pd.read_parquet(ref_p)
    ref = ref[(ref["method"] == "mclp_place")]

    ks = sorted(set(ctl["K"].unique()))
    ok = True
    for k in ks:
        a = _last10(ctl[ctl["K"] == k])
        b = _last10(ref[ref["K"] == k])
        shared = sorted(set(a.index) & set(b.index))
        if not shared:
            print(f"[v6-control] K={k}: no shared seeds between control and v5")
            ok = False
            continue
        d = (a.loc[shared] - b.loc[shared]).abs()
        worst = float(d.max())
        status = "OK" if worst <= TOL else "MISMATCH"
        print(
            f"[v6-control] K={k:<3} seeds={len(shared)}  max|diff|={worst:.3e}  {status}"
        )
        if worst > TOL:
            ok = False
            for s in shared:
                if abs(a[s] - b[s]) > TOL:
                    print(f"              seed {s}: control {a[s]:.6f} vs v5 {b[s]:.6f}")

    if not ok:
        print(
            "[v6-control] FAILED — switches-off no longer reproduces v5. The C1/C2 "
            "edits changed the baseline, so the v6 arms would measure the refactor."
        )
        return 1
    print("[v6-control] control reproduces v5 exactly — v6 arms are interpretable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
