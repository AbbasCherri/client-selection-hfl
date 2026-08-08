"""Fleet-size sweep: (K, capacity) pairing, job wiring, and the shipped config.

The sweep exists to ask whether placement matters more with fewer UAVs. Two
things would silently ruin it:

  1. **K not actually applied per job.** If ``fl.K``/``fl.capacity`` failed to
     reach ``run_full_hfl``, every cell would run the same fleet and the sweep
     would produce a flat, entirely fake negative result.
  2. **Total slots drifting across cells.** Capacity is swept *with* K to hold
     ``K*C`` constant. If that pairing breaks, small-K cells starve the model of
     data, and starvation is indistinguishable from placement irrelevance in
     macro-F1 — the same trap that made the 20 km class-realism placement
     contrast powerless.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _fixture import build_synthetic_raw  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.sweep import run_uav_sweep  # noqa: E402
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


def shipped_config_holds_slots_constant():
    """paper_uav_count.yaml must keep K*C fixed, or the sweep is confounded."""
    cfg = load_config(_REPO / "configs" / "paper_uav_count.yaml")
    pairs = [(int(e["K"]), int(e["capacity"])) for e in cfg["K_values"]]
    slots = [k * c for k, c in pairs]
    assert len(set(slots)) == 1, f"total slots K*C must be constant, got {slots} for {pairs}"
    assert slots[0] == 120, f"expected 120 slots (matching paper_full), got {slots[0]}"
    # The point of the sweep is the small-K regime; without it there is no
    # unsaturated cell and the whole run repeats the coverage sweep's blind spot.
    assert min(k for k, _ in pairs) <= 3, f"no small-K cell: {pairs}"
    assert "flat_fl" in cfg["methods"], "flat_fl is the K-invariance validity check"
    assert "random_place" in cfg["methods"], "random_place is the placement floor"


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
check("shipped config holds K*C constant", shipped_config_holds_slots_constant)
check("flat_fl is invariant to K", flat_fl_is_k_invariant)
finish()
