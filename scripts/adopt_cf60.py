"""Emit configs/tuned_weights_cf60.yaml from the cf60_* tuning studies.

Adoption rule, applied identically to all eight methods: the trial that ranks
FIRST on the 3 tuning seeds. The 2 transfer seeds are reported but never used to
select.

Why not pooled val, which `configs/tuned_weights.yaml` used for fair_mab and
power_of_choice:

  1. Selecting on the transfer seeds spends them. Their whole function is to
     answer "does the tune-set winner hold up on data the search never saw",
     and a config chosen partly on those seeds can no longer answer it.
  2. Run over these eight studies, the pooled rule moves the pick off tune-rank
     1 for EXACTLY ONE method -- proposed_hfl, the method under test, in its own
     favour by +0.0046 pooled. A rule that fires once and only for the home team
     is indefensible at review no matter how mechanical its derivation.

Tune-rank 1 costs proposed_hfl 0.0046 of pooled val relative to the alternative.
That is the conservative direction, which is where a contested claim belongs.

The script prints both so the choice is auditable rather than buried.

Base `fl:` recipe = proposed_hfl's adopted recipe, matching the convention that
the ablations (hfl_static, hfl_no_reputation) differ from the proposed system
only in the ablated component and not in their training recipe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HPO = Path("results/hpo_5km")
METHODS = ("proposed_hfl", "fedcs", "oort", "flat_fl", "hfl_no_selection",
           "rep_cap", "fair_mab", "power_of_choice")
PARAMS = ("lr", "lr_decay", "logit_adjust_tau", "server_momentum", "ema_decay")
N_TUNE, N_TRANSFER = 3, 2


def pick(method: str):
    lb = pd.read_csv(HPO / f"cf60_{method}_leaderboard.csv")
    tr = pd.read_csv(HPO / f"cf60_{method}_transfer.csv")
    tr["pooled"] = (N_TUNE * tr["val_tune"] + N_TRANSFER * tr["val_transfer"]) / (
        N_TUNE + N_TRANSFER)
    best = tr[tr["tune_rank"] == 1].iloc[0]
    alt = tr.sort_values("pooled", ascending=False).iloc[0]
    row = lb[lb["number"] == best["trial"]].iloc[0]
    params = {p: row[f"params_{p}"] for p in PARAMS}
    return {
        "trial": int(best["trial"]),
        "tune_rank": int(best["tune_rank"]),
        "val_tune": float(best["val_tune"]),
        "val_transfer": float(best["val_transfer"]),
        "pooled": float(best["pooled"]),
        "alt_trial": int(alt["trial"]),
        "alt_rank": int(alt["tune_rank"]),
        "alt_pooled": float(alt["pooled"]),
        "params": params,
    }


def fmt(params: dict, indent: str) -> list[str]:
    out = [f"{indent}lr: {params['lr']:.6e}",
           f"{indent}uav_lr: {params['lr']:.6e}",
           f"{indent}lr_decay: {params['lr_decay']}",
           f"{indent}logit_adjust_tau: {params['logit_adjust_tau']:.6f}",
           f"{indent}server_momentum: {params['server_momentum']:.6f}",
           f"{indent}ema_decay: {params['ema_decay']:.6f}"]
    return out


def main() -> int:
    chosen = {m: pick(m) for m in METHODS}

    print("adoption: tune-rank 1 for every method; transfer seeds NOT used to select")
    print(f"{'method':>18} {'trial':>6} {'tune':>8} {'xfer':>8} {'pooled':>8}  "
          f"{'pooled-rule would pick':>24}")
    moved = []
    for m, c in chosen.items():
        note = "same"
        if c["alt_trial"] != c["trial"]:
            moved.append(m)
            note = (f"trial {c['alt_trial']} (rank {c['alt_rank']}), "
                    f"{c['alt_pooled'] - c['pooled']:+.4f}")
        print(f"{m:>18} {c['trial']:>6} {c['val_tune']:>8.4f} "
              f"{c['val_transfer']:>8.4f} {c['pooled']:>8.4f}  {note:>24}")
    print()
    print(f"pooled rule would have differed for: {moved or 'no method'}")
    print("declined: it spends the held-out check and fires only for the method "
          "under test")

    base = chosen["proposed_hfl"]
    lines = [
        "# Validation-tuned recipes — 60 ROUNDS, `fusion_owner: client_full`.",
        "#",
        "# Supersedes configs/tuned_weights.yaml FOR client_full experiments only.",
        "# Inherit with `extends: tuned_weights_cf60.yaml`; never copy these values.",
        "#",
        "# Why this file exists",
        "# --------------------",
        "# The 2026-08-05 recipes in tuned_weights.yaml were searched at 20 rounds",
        "# under `fusion_owner: uav`. Two measurements invalidate them for the",
        "# client_full table (both recorded in PREREGISTRATION.md, 2026-09-05):",
        "#",
        "#   1. Rank correlation between a 20-round ordering and the final",
        "#      ordering is rho = +0.22. At 60 rounds it is +0.97.",
        "#   2. Applying those per-method recipes made fedcs WORSE by up to 0.09",
        "#      and oort by up to 0.055 versus not tuning per method at all, and",
        "#      the sign of the headline comparison flipped with the regime.",
        "#",
        "# Search, identical for every method: --space recipe, 25 trials, 60 rounds,",
        "# tuning seeds 20-22, N in {30,50}, R_comm 5000, K=20, capacity 6,",
        "# subsample 0.2, objective val_macro_f1. Equal budget, equal space, equal",
        "# horizon, equal seeds — an asymmetric budget is the confound this replaces.",
        "#",
        "# proposed_hfl was searched on the RECIPE space only, like everyone else;",
        "# its fitness weights are held at their existing values. That is",
        "# conservative for our claim, not favourable to it.",
        "#",
        "# Adoption: tune-rank 1 on the 3 tuning seeds, for every method. The 2",
        "# transfer seeds are reported and never used to select, so the transfer",
        "# check still answers the question it exists to answer. Applied by script",
        "# (scripts/adopt_cf60.py) to all eight studies, not by hand.",
        "#",
        "# tuned_weights.yaml used POOLED val (3*tune + 2*transfer)/5 for two",
        "# methods. That rule is declined here: over these eight studies it moves",
        "# the pick for exactly one method — proposed_hfl, the method under test,",
        "# in its own favour by +0.0046 pooled. Tune-rank 1 costs us that 0.0046,",
        "# which is the correct direction for a contested claim.",
        "#",
        "#      method          trial  tune-rank    tune    xfer  pooled",
    ]
    for m, c in chosen.items():
        lines.append(f"#   {m:>18} {c['trial']:>6} {c['tune_rank']:>10} "
                     f"{c['val_tune']:>7.4f} {c['val_transfer']:>7.4f} "
                     f"{c['pooled']:>7.4f}")
    lines += [
        "#",
        f"# The declined pooled rule would have differed for: "
        f"{', '.join(moved) if moved else 'no method'}.",
        "# Stated plainly because it is the kind of detail that decides whether a",
        "# reviewer trusts the table.",
        "",
        "extends: paper_full.yaml",
        "",
        "fl:",
        "  # Base recipe = proposed_hfl's adopted recipe. hfl_static and",
        "  # hfl_no_reputation inherit it so they ablate one component, not the",
        "  # recipe as well. centralized and mozaffari/alzenad also inherit it.",
    ]
    lines += fmt(base["params"], "  ")
    lines += ["", "  per_method:"]
    for m in METHODS:
        if m == "proposed_hfl":
            continue
        c = chosen[m]
        lines.append(f"    {m}:   # trial {c['trial']}, tune-rank {c['tune_rank']}, "
                     f"pooled {c['pooled']:.4f}")
        lines += fmt(c["params"], "      ")
    lines.append("")

    out = Path("configs/tuned_weights_cf60.yaml")
    out.write_text("\n".join(lines))
    print(f"\nwrote {out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
