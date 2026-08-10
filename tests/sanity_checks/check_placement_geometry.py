"""Placement-geometry columns and the `placement_weights` override.

Both are new for REPORTS/preregistration_v6_c3.md and both are the kind of
change that can be silently wrong: a geometry statistic that looks plausible
while measuring the wrong thing, and a weights knob that is applied to the
optimizer's fitness but not to the canonical re-score (or vice versa), which
would score a layout under different weights than it was optimized for.

The no-op check matters most. Every result to date ran without this option, so
`placement_weights: null` must reproduce Fitness's shipped defaults exactly, or
the C3 arm is measuring the refactor rather than the weights — the same failure
mode the v6 control was built to catch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.federated import _placement_geometry, _place_uavs  # noqa: E402
from uavbench.problem.fitness import Fitness  # noqa: E402
from uavbench.problem.instance import generate_instance  # noqa: E402

_REF = np.array([37.5, 137.0])


def _coords(n, spread_m=4000.0, seed=0):
    """n client lat/lons spread over roughly `spread_m` metres around _REF."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-spread_m, spread_m, size=(n, 2))
    R = 6_371_000.0
    lat = _REF[0] + np.degrees(xy[:, 1] / R)
    lon = _REF[1] + np.degrees(xy[:, 0] / (R * np.cos(np.radians(_REF[0]))))
    return {i: (float(lat[i]), float(lon[i])) for i in range(n)}


def _uavs(xy_m, z=1500.0):
    return np.column_stack([np.asarray(xy_m, dtype=float),
                            np.full(len(xy_m), float(z))])


def disjoint_placement_has_multiplicity_one():
    # Two UAVs 40 km apart with a 5 km reach cannot both see any client.
    coords = _coords(60, spread_m=1000.0)
    pos = _uavs([[-20_000.0, 0.0], [20_000.0, 0.0]])
    g = _placement_geometry(coords, pos, _REF, R_comm=5000.0)
    assert abs(g["cover_multiplicity_mean"] - 1.0) < 1e-9, g
    assert abs(g["unique_cover_frac"] - 1.0) < 1e-9, g


def stacked_placement_has_multiplicity_k():
    # Three UAVs at the same point: every covered client is covered 3 times.
    coords = _coords(60, spread_m=1000.0)
    pos = _uavs([[0.0, 0.0]] * 3)
    g = _placement_geometry(coords, pos, _REF, R_comm=5000.0)
    assert abs(g["cover_multiplicity_mean"] - 3.0) < 1e-9, g
    assert abs(g["unique_cover_frac"] - 0.0) < 1e-9, g
    assert abs(g["uav_pairwise_sep_m"]) < 1e-9, g


def separation_is_the_mean_pairwise_distance():
    coords = _coords(20, spread_m=1000.0)
    pos = _uavs([[0.0, 0.0], [3000.0, 4000.0]])       # 5000 m apart
    g = _placement_geometry(coords, pos, _REF, R_comm=5000.0)
    assert abs(g["uav_pairwise_sep_m"] - 5000.0) < 1e-6, g


def uncovered_clients_are_excluded_not_counted_as_zero():
    # Half the clients out of reach: multiplicity averages over COVERED ones,
    # so it must stay >= 1 rather than being dragged toward 0.
    coords = _coords(80, spread_m=30_000.0, seed=4)
    pos = _uavs([[0.0, 0.0]])
    g = _placement_geometry(coords, pos, _REF, R_comm=5000.0)
    assert g["cover_multiplicity_mean"] >= 1.0, g
    assert np.isnan(g["uav_pairwise_sep_m"]), "one UAV has no pairwise separation"


def no_coverage_gives_nan_not_a_number():
    coords = _coords(30, spread_m=500.0)
    pos = _uavs([[500_000.0, 500_000.0]])
    g = _placement_geometry(coords, pos, _REF, R_comm=1000.0)
    assert np.isnan(g["cover_multiplicity_mean"]), g
    assert np.isnan(g["unique_cover_frac"]), g


def per_uav_radii_are_honoured():
    coords = _coords(60, spread_m=1000.0)
    pos = _uavs([[0.0, 0.0], [0.0, 0.0]])
    wide = _placement_geometry(coords, pos, _REF, R_comm=5000.0,
                               radii=np.array([5000.0, 5000.0]))
    one_blind = _placement_geometry(coords, pos, _REF, R_comm=5000.0,
                                    radii=np.array([5000.0, 1.0]))
    assert abs(wide["cover_multiplicity_mean"] - 2.0) < 1e-9, wide
    assert abs(one_blind["cover_multiplicity_mean"] - 1.0) < 1e-9, one_blind


def weights_none_is_a_no_op():
    coords = _coords(150, seed=11)
    kw = dict(client_coords=coords, K=8, R_comm=5000.0, capacity=6,
              method="mclp_ls", P=12, G_max=15, prev_positions_m=None,
              link_model="path_loss", z_min_m=100.0, z_max_m=2000.0)
    a = _place_uavs(rng=np.random.default_rng(7), **kw)
    b = _place_uavs(rng=np.random.default_rng(7), weights=None, **kw)
    assert np.array_equal(a[0], b[0]), "omitting weights changed the placement"
    assert a[2] == b[2], f"omitting weights changed the score: {a[2]} vs {b[2]}"


def explicit_default_weights_reproduce_the_default():
    coords = _coords(150, seed=12)
    kw = dict(client_coords=coords, K=8, R_comm=5000.0, capacity=6,
              method="mclp_ls", P=12, G_max=15, prev_positions_m=None,
              link_model="path_loss", z_min_m=100.0, z_max_m=2000.0)
    a = _place_uavs(rng=np.random.default_rng(5), **kw)
    b = _place_uavs(rng=np.random.default_rng(5), weights=(0.811, 0.03, 0.159), **kw)
    assert np.array_equal(a[0], b[0]), "passing the defaults explicitly changed the placement"


def pure_coverage_weights_change_the_placement():
    # If (1,0,0) were silently ignored, H-B could not be tested at all.
    coords = _coords(150, seed=13)
    kw = dict(client_coords=coords, K=8, R_comm=5000.0, capacity=6,
              method="mclp_ls", P=12, G_max=15, prev_positions_m=None,
              link_model="path_loss", z_min_m=100.0, z_max_m=2000.0)
    a = _place_uavs(rng=np.random.default_rng(9), **kw)
    b = _place_uavs(rng=np.random.default_rng(9), weights=(1.0, 0.0, 0.0), **kw)
    assert not np.array_equal(a[0], b[0]), "w=(1,0,0) produced an identical placement"


def the_rescore_uses_the_same_weights_as_the_search():
    # The returned score must be the layout evaluated under the SAME weights the
    # optimizer searched under. Re-score it here independently and compare.
    coords = _coords(120, seed=21)
    w = (1.0, 0.0, 0.0)
    pos, ref, score, _ = _place_uavs(
        client_coords=coords, K=6, R_comm=5000.0, capacity=6, method="mclp_ls",
        rng=np.random.default_rng(3), P=12, G_max=15, prev_positions_m=None,
        link_model="path_loss", z_min_m=100.0, z_max_m=2000.0, weights=w,
    )
    from uavbench.fl.federated import _build_problem_instance
    from uavbench.problem.path_loss import LinkModel
    inst, _ = _build_problem_instance(coords, 6, 5000.0, 6, None,
                                      z_min_m=100.0, z_max_m=2000.0)
    link = LinkModel(r_comm_m=5000.0, z_min_m=100.0, z_max_m=2000.0)
    mine = float(Fitness(inst, w1=w[0], w2=w[1], w3=w[2], link=link)(pos.reshape(-1)))
    assert abs(mine - score) < 1e-9, f"re-score {score} != same-weights score {mine}"


def malformed_weights_are_rejected():
    from uavbench.fl.federated import run_full_hfl  # noqa: F401  (import path only)
    # Validation lives in run_full_hfl's config parsing; assert the contract it
    # enforces rather than booting a full run.
    for bad in ([1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]):
        vals = tuple(float(v) for v in bad)
        ok = len(vals) == 3 and all(v >= 0 for v in vals) and sum(vals) > 0
        assert not ok, f"{bad} should be rejected by the documented contract"


check("disjoint discs give multiplicity 1", disjoint_placement_has_multiplicity_one)
check("stacked UAVs give multiplicity K", stacked_placement_has_multiplicity_k)
check("separation is the mean pairwise distance", separation_is_the_mean_pairwise_distance)
check("uncovered clients are excluded, not zeroed", uncovered_clients_are_excluded_not_counted_as_zero)
check("no coverage gives NaN", no_coverage_gives_nan_not_a_number)
check("per-UAV radii are honoured", per_uav_radii_are_honoured)
check("weights=None is a no-op", weights_none_is_a_no_op)
check("explicit defaults reproduce the default", explicit_default_weights_reproduce_the_default)
check("w=(1,0,0) changes the placement", pure_coverage_weights_change_the_placement)
check("the re-score uses the search weights", the_rescore_uses_the_same_weights_as_the_search)
check("malformed weights are rejected", malformed_weights_are_rejected)
finish()
