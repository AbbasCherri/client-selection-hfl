"""Fleet-size sweep: (K, capacity) pairing, job wiring, and the shipped config.

The sweep exists to ask whether placement matters more with fewer UAVs. Two
things would silently ruin it:

  1. **K not actually applied per job.** If ``fl.K``/``fl.capacity`` failed to
     reach ``run_full_hfl``, every cell would run the same fleet and the sweep
     would produce a flat, entirely fake negative result.
  2. **Per-UAV capacity drifting across cells.** Capacity is held fixed and
     above the level where a UAV's pooled shard goes single-class and the run
     forgets what it learned (measured: cap<=3 all end near the constant-
     predictor floor). If it drifts, cells stop differing only in fleet size and
     start differing in whether they can learn at all — and that is
     indistinguishable from placement irrelevance in macro-F1.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _fixture import build_synthetic_raw  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.sweep import run_uav_sweep  # noqa: E402
from uavbench.reporting import build_seed_manifest  # noqa: E402
from uavbench.runner import load_config  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]


def _cfg(results_dir, pairs):
    return {
        "results_dir": results_dir,
        "N": 12,
        "K_values": pairs,
        "methods": ["proposed_hfl"],
        "n_seeds": 1,
        "n_workers": 1,
        "optimizer_seed": 42,
        "fl": {
            "n_rounds": 1, "n_local_epochs": 1, "n_uav_epochs": 1,
            "lr": 0.01, "uav_lr": 0.01, "batch_size": 4, "K": 99,
            "R_comm": 200_000.0, "capacity": 99, "T_sel": 1,
            "lambda_min": 0.0, "target_accuracy": 0.99, "seed": 42,
        },
        "budget": {"P": 5, "G_max": 3},
        "data": {
            "source": "prebuilt",
            "prebuilt": build_synthetic_raw(N=12, K=2, seed=42),
            "seed": 42,
        },
    }


def k_and_capacity_reach_the_run():
    """Each cell must actually run its own fleet size, not the config default."""
    pairs = [{"K": 2, "capacity": 6}, {"K": 4, "capacity": 3}]
    with tempfile.TemporaryDirectory() as d:
        df = run_uav_sweep(_cfg(d, pairs))["rounds"]
        assert {"K", "capacity"} <= set(df.columns), f"missing K/capacity columns: {df.columns}"
        assert sorted(df["K"].unique()) == [2, 4], f"K column: {sorted(df['K'].unique())}"
        # The resolved per-job config is the ground truth — the column could be
        # right while the run used fl.K=99 from the template.
        for k, cap in ((2, 6), (4, 3)):
            resolved = Path(d) / f"K{k}" / "seed0" / "proposed_hfl" / "config.fullsim.resolved.yaml"
            assert resolved.exists(), f"no resolved config for K={k}"
            text = resolved.read_text()
            assert f"K: {k}" in text, f"K={k} cell did not receive fl.K={k}"
            assert f"capacity: {cap}" in text, f"K={k} cell did not receive capacity={cap}"
            assert "K: 99" not in text, f"K={k} cell kept the template's placeholder fleet"


def malformed_pairs_are_rejected():
    """A bare int list would silently lose the capacity pairing."""
    for bad in ([2, 4], [{"K": 2}], [{"capacity": 6}]):
        with tempfile.TemporaryDirectory() as d:
            try:
                run_uav_sweep(_cfg(d, bad))
            except ValueError:
                continue
            raise AssertionError(f"K_values={bad!r} was accepted")


def shipped_config_holds_capacity_constant():
    """paper_uav_count.yaml must keep per-UAV capacity fixed and above the floor.

    This asserts the OPPOSITE of what it asserted before 2026-08-09, and the
    reversal is the point. The old invariant was K*C = 120, which pairs large
    fleets with tiny capacities; results/probe_topology then measured that
    capacity, not coverage, decides whether the pipeline learns, and that the
    whole high-K half of such a grid (cap 1-3) forgets by round 10. A grid built
    on the old invariant would report capacity-induced collapse as a fleet-size
    effect on placement.

    So capacity is now the constant and total slots vary. The confound moves
    from "does this cell learn at all" to "fleet size is entangled with data
    volume", which is the lesser of the two: the per-K placement contrasts stay
    internally controlled either way, and only the cross-K trend is affected.
    """
    cfg = load_config(_REPO / "configs" / "paper_uav_count.yaml")
    pairs = [(int(e["K"]), int(e["capacity"])) for e in cfg["K_values"]]
    caps = {c for _, c in pairs}
    assert len(caps) == 1, (
        f"per-UAV capacity must be constant across the grid, got {sorted(caps)} for {pairs} — "
        "varying it makes cells differ in whether they can learn, not just in fleet size"
    )
    cap = caps.pop()
    # 4 is where recovery appeared in the probe (cap=3 ended at 0.286, cap=4 at
    # 0.379). Asserted with margin rather than at the measured edge: the floor
    # was measured at one N and one radius and should not be treated as exact.
    assert cap >= 4, (
        f"capacity {cap} is at or below the level where UAV shards go single-class "
        "and the run unlearns (probe_topology: cap<=3 all ended near the "
        "constant-predictor floor)"
    )
    assert cap <= 6, f"per-UAV capacity {cap} is not operationally plausible"

    # The grid must reach genuinely under-covered fleets, because coverage is
    # flat (82-89%) from K=20 upward at this radius — cells above 20 differ in
    # aircraft count without differing in reach, so a grid living entirely up
    # there cannot show a placement effect at all.
    ks = sorted(k for k, _ in pairs)
    assert ks[0] <= 10, f"no under-covered cell — smallest K is {ks[0]}, need <= 10"
    assert 20 in ks, "grid must include K=20, the paper operating point, as its anchor"

    # Selection must still bind in every cell, or the top of the grid quietly
    # stops testing the selection rule: with ~185 clients covered, a slot budget
    # above that admits everyone and the comparison changes meaning mid-sweep.
    slots = [k * cap for k, _ in pairs]
    assert max(slots) <= 185, (
        f"largest cell has {max(slots)} slots against ~185 covered clients — "
        "selection stops binding there"
    )
    assert "flat_fl" in cfg["methods"], "flat_fl is the K-invariance validity check"
    assert "random_place" in cfg["methods"], "random_place is the placement floor"


def seed_manifest_covers_the_uav_harness():
    """The CLI writes a seed manifest before running — that path must work.

    Caught late: the checks above call run_uav_sweep directly, so they skipped
    _write_seed_manifest entirely and the sweep died on its first real CLI
    invocation with `unknown harness 'uav'`. Testing the library function while
    the entry point is broken is exactly the gap this closes.
    """
    cfg = load_config(_REPO / "configs" / "paper_uav_count.yaml")
    man = build_seed_manifest(cfg, "uav")
    expected = len(cfg["K_values"]) * len(cfg["methods"]) * cfg.get("n_seeds", 1)
    assert len(man) == expected, f"manifest has {len(man)} rows, expected {expected}"
    for col in ("K", "capacity", "seed", "partition_seed"):
        assert col in man.columns, f"manifest missing {col}"
    # Pairing across fleet sizes requires the same seed at each K.
    per_k = man.groupby("K")["seed"].apply(lambda s: sorted(s.unique()))
    first = per_k.iloc[0]
    for k, seeds in per_k.items():
        assert seeds == first, f"K={k} draws different seeds — cells would not be paired"


def flat_fl_is_k_invariant():
    """flat_fl has no UAVs — if it moves with K, something other than placement is."""
    pairs = [{"K": 2, "capacity": 6}, {"K": 4, "capacity": 3}]
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(d, pairs)
        cfg["methods"] = ["flat_fl"]
        df = run_uav_sweep(cfg)["rounds"]
        by_k = df.groupby("K")["macro_f1"].apply(list)
        ref = by_k.iloc[0]
        for k, vals in by_k.items():
            assert np.allclose(vals, ref), f"flat_fl moved with K={k}: {vals} vs {ref}"


check("K and capacity reach each job", k_and_capacity_reach_the_run)
check("malformed K_values are rejected", malformed_pairs_are_rejected)
check("shipped config holds capacity constant, above the floor",
      shipped_config_holds_capacity_constant)
check("seed manifest covers the uav harness", seed_manifest_covers_the_uav_harness)
check("flat_fl is invariant to K", flat_fl_is_k_invariant)
finish()
