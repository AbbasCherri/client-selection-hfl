"""Tier-1 must actually be scored through the air-to-ground channel.

Tier-1 is the paper's headline placement table, and until 2026-08-09 it ran on a
flat ``slant_distance <= R_comm`` sphere. Under that gate altitude can only ever
*add* slant distance, so every optimizer drives z to the floor: the benchmark
advertised a 3D search whose third dimension had a known closed-form answer.

Wiring a link model into the runner is a two-part change and both parts can fail
silently. The optimizer can be scored through the channel while
``compute_metrics`` rebuilds a *fresh* Fitness without it — in which case every
reported coverage number is flat-gate coverage for positions chosen under the
channel. And ``Fitness`` lets an explicit ``radii`` argument override the link,
so forwarding a baseline's own ``meta["radii"]`` re-creates the unequal-radius
comparison the shared channel was meant to retire.

So these checks assert behaviour end-to-end through ``_run_one``, not that the
config file contains the right string.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import copy  # noqa: E402

import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.metrics.placement import compute_metrics  # noqa: E402
from uavbench.optimizers.base import Result  # noqa: E402
from uavbench.problem.instance import generate_instance  # noqa: E402
from uavbench.problem.link import LinkModel  # noqa: E402
from uavbench.runner import _run_one, load_config  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]


def _tiny_cfg(link_model: str) -> dict:
    """tier1_core with one small scenario and a token budget."""
    cfg = copy.deepcopy(load_config(_REPO / "configs" / "tier1_core.yaml"))
    cfg["scenarios"] = [{"distribution": "uniform", "N": 40, "K": 4}]
    cfg["budget"] = {"P": 8, "G_max": 5}
    cfg["problem"]["link_model"] = link_model
    cfg["problem"]["capacity"] = 15
    return cfg


def _instance(n: int = 400, k: int = 4):
    cfg = load_config(_REPO / "configs" / "tier1_core.yaml")
    inst = generate_instance(
        distribution="uniform", N=n, K=k, area=cfg["area"], seed=7,
        capacity=n, uav_battery=1.0, R_comm=float(cfg["problem"]["R_comm"]),
        B_min_uav=0.2, beta_mode="pinned", t=0, T_decay=20,
    )
    return cfg, inst


def reported_coverage_is_computed_through_the_link():
    """compute_metrics must honour the link, or every reported number is wrong.

    The failure this catches is silent: the optimizer searches under the channel,
    the metrics rebuild a fresh Fitness without it, and the table reports
    flat-gate coverage for channel-chosen positions. The two agree only where
    altitude does not matter — i.e. nowhere interesting.
    """
    cfg, inst = _instance()
    z_lo, z_hi = (float(v) for v in cfg["area"]["z"])
    link = LinkModel(r_comm_m=float(cfg["problem"]["R_comm"]), z_min_m=z_lo, z_max_m=z_hi)

    # Park every UAV at the FLOOR. The channel's ground radius there is roughly
    # z_lo/tan(theta_opt) ~ 270 m against the flat gate's 500 m, a ~3x area gap
    # — the largest disagreement available inside the band. (The ceiling is the
    # wrong probe: sqrt(r^2 + z^2) at 400 m actually exceeds the flat 500 m
    # gate, so the two models nearly agree there and the test would pass
    # vacuously.) Capacity is set to N so assignment limits cannot mask the
    # radius difference.
    K = inst.K
    pos = np.column_stack([inst.device_coords[:K, :2], np.full(K, z_lo)])
    res = Result(method="probe", best_position=pos.ravel(), best_fitness=0.0,
                 convergence=[0.0], meta={})

    fw = (cfg["fitness"]["w1"], cfg["fitness"]["w2"], cfg["fitness"]["w3"])
    flat = compute_metrics(inst, res, fitness_weights=fw, link=None)
    chan = compute_metrics(inst, res, fitness_weights=fw, link=link)
    assert chan["coverage_pct"] < flat["coverage_pct"], (
        f"channel coverage {chan['coverage_pct']:.2f}% is not below flat-gate "
        f"{flat['coverage_pct']:.2f}% at the altitude floor — compute_metrics is not "
        "applying the link, so Tier-1 would report flat-gate coverage for "
        "positions chosen under the channel"
    )


def the_runner_scores_the_two_link_models_differently():
    """End-to-end: the config switch must reach the reported metrics."""
    a = _run_one(_tiny_cfg("path_loss"), "pso", 0, 0, 0)["metrics"]
    b = _run_one(_tiny_cfg("range_gate"), "pso", 0, 0, 0)["metrics"]
    assert a["coverage_pct"] != b["coverage_pct"] or a["final_fitness"] != b["final_fitness"], (
        f"path_loss and range_gate produced identical Tier-1 metrics "
        f"(coverage {a['coverage_pct']}%, fitness {a['final_fitness']}) — the "
        "link_model setting is not reaching the runner"
    )


def an_unknown_link_model_is_rejected():
    """A typo must fail loudly rather than silently falling back to the flat gate."""
    cfg = _tiny_cfg("pathloss")  # missing underscore
    try:
        _run_one(cfg, "pso", 0, 0, 0)
    except ValueError as e:
        assert "link_model" in str(e), f"wrong error: {e}"
        return
    raise AssertionError("an unknown link_model was accepted and silently ignored")


def altitude_is_not_pinned_to_a_bound_under_the_channel():
    """The point of the whole change: z must become a real decision.

    Asserted on the mean rather than per-UAV — a single UAV may legitimately sit
    at a bound. If the mean sits within a metre of either bound, the search has
    collapsed back to 2D.
    """
    cfg = _tiny_cfg("path_loss")
    z_lo, z_hi = (float(v) for v in cfg["area"]["z"])
    out = _run_one(cfg, "pso", 0, 0, 0)["metrics"]
    # Must come from the run. An earlier version of this check fell back to
    # LinkModel.z_star_m when the metric was missing, which tested the channel's
    # analytic optimum rather than what the optimizer actually did — it passed
    # while the runner was still on the flat gate.
    assert "mean_altitude_m" in out, (
        "mean_altitude_m is not reported — without it this check cannot see "
        "where the optimizer put the fleet"
    )
    z_mean = out["mean_altitude_m"]
    assert z_lo + 1.0 < z_mean < z_hi - 1.0, (
        f"mean altitude {z_mean:.1f} m is pinned at a bound of [{z_lo}, {z_hi}] — "
        "the vertical decision is degenerate"
    )


def the_flat_gate_still_pins_altitude_low():
    """Guard the guard: the interior-altitude check must be able to fail.

    Under the range gate altitude is a pure penalty, so the fleet should sit
    markedly lower than under the channel. If this stops holding, the check
    above proves nothing.
    """
    cfg = _tiny_cfg("range_gate")
    z_lo, z_hi = (float(v) for v in cfg["area"]["z"])
    flat = _run_one(cfg, "pso", 0, 0, 0)["metrics"]["mean_altitude_m"]
    chan = _run_one(_tiny_cfg("path_loss"), "pso", 0, 0, 0)["metrics"]["mean_altitude_m"]
    assert flat < chan, (
        f"flat-gate mean altitude {flat:.1f} m is not below the channel's {chan:.1f} m — "
        "the two link models are not producing different vertical behaviour, so "
        "'altitude is interior' is not evidence the channel is being applied"
    )


check("reported coverage is computed through the link",
      reported_coverage_is_computed_through_the_link)
check("the runner scores the two link models differently",
      the_runner_scores_the_two_link_models_differently)
check("an unknown link_model is rejected", an_unknown_link_model_is_rejected)
check("altitude is not pinned to a bound under the channel",
      altitude_is_not_pinned_to_a_bound_under_the_channel)
check("the flat gate still pins altitude low", the_flat_gate_still_pins_altitude_low)
finish()
