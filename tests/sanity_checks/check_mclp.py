"""MCLP near-optimality reference invariants.

The paper's "PSO reaches X% of the MILP-optimal coverage" claim rests entirely
on this solver being (a) actually optimal and (b) measuring the same thing the
heuristics are measured on. Both were violated through 2026-07:

  * coverage used 2-D ground range while ``ProblemInstance.distances`` (and so
    every heuristic) uses the 3-D slant distance;
  * ``covered_value_norm`` divided a subsampled client set's covered value by
    the *full* instance total;
  * a bare ``optimal`` boolean hid whether the time limit bound first.

The load-bearing property is the dominance check: an exact max-covering optimum
must be >= the coverage any heuristic achieves on the same instance, at the same
radius, on the grid. If that ever fails, the "optimum" is not one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.problem.exact import mclp_reference  # noqa: E402
from uavbench.problem.fitness import Fitness  # noqa: E402
from uavbench.problem.instance import generate_instance  # noqa: E402

AREA = {"x": [0.0, 2000.0], "y": [0.0, 2000.0], "z": [20.0, 120.0]}


def _toy(seed: int = 3, N: int = 40, K: int = 3, R: float = 500.0):
    return generate_instance(
        distribution="clustered", N=N, K=K, area=AREA, seed=seed,
        capacity=15, uav_battery=1.0, R_comm=R, B_min_uav=0.2,
        beta_mode="pinned", t=0, T_decay=20, prev_mode="stale",
    )


def optimum_dominates_a_heuristic():
    """The MILP optimum must beat any placement we can construct by hand."""
    inst = _toy()
    ref = mclp_reference(inst, grid_res=12, time_limit=120.0)
    fit = Fitness(inst, w1=1.0, w2=0.0, w3=0.0)  # pure coverage

    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(40):
        pos = np.column_stack([
            rng.uniform(AREA["x"][0], AREA["x"][1], inst.K),
            rng.uniform(AREA["y"][0], AREA["y"][1], inst.K),
            np.full(inst.K, 0.5 * (AREA["z"][0] + AREA["z"][1])),
        ])
        best = max(best, fit.components(pos.ravel()).f_cover_norm)

    assert ref.covered_value_norm >= best - 1e-9, (
        f"MCLP 'optimum' {ref.covered_value_norm:.4f} lost to a random "
        f"heuristic {best:.4f} — it is not an optimum"
    )


def cip_optimum_dominates_the_actual_optimizers():
    """The bound must beat the methods it bounds — random placements are too weak.

    ``optimum_dominates_a_heuristic`` compares against 40 uniform random layouts
    on a K=3 instance, which any formulation clears. It passed while the
    capacitated reference was reporting **70% of optimum against methods reaching
    100%**: dominance-pruned sites plus a binary "site used" variable forbade
    stationing two UAVs on one over-capacity cluster. This checks the bound where
    it actually failed — against mclp_ls and pso, at a K and capacity that bind.
    """
    from uavbench.optimizers import build_optimizer

    for seed in (0, 1, 2):
        inst = _toy(seed=seed, N=60, K=6, R=1000.0)
        ref = mclp_reference(inst, sites="cip", time_limit=180.0)
        assert ref.optimal, f"[seed {seed}] reference did not solve to optimality"
        fit_pure = Fitness(inst, w1=1.0, w2=0.0, w3=0.0)
        for name in ("mclp_ls", "pso"):
            got = build_optimizer(name, {}, {"P": 60, "G_max": 40}).optimize(
                inst, Fitness(inst, w1=1.0, w2=0.0, w3=0.0), np.random.default_rng(5)
            )
            cov = fit_pure.components(got.best_position).f_cover_norm
            assert ref.covered_value_norm >= cov - 1e-6, (
                f"[seed {seed}] MCLP 'optimum' {ref.covered_value_norm:.4f} is below "
                f"{name}'s {cov:.4f} ({100 * cov / max(ref.covered_value_norm, 1e-9):.1f}% "
                "of 'optimum') — the reference is not an upper bound"
            )


def cip_reference_is_at_least_the_grid_reference():
    """Church's set contains a planar optimum, so it cannot lose to a grid.

    The circle-intersection candidates provably contain an optimal placement over
    the *whole plane*, which includes every grid point. A CIP reference scoring
    below the grid one therefore means the candidate construction or the pruning
    dropped something it should not have.
    """
    for seed in (0, 1):
        inst = _toy(seed=seed, N=50, K=4, R=600.0)
        cip = mclp_reference(inst, sites="cip", time_limit=180.0)
        grid = mclp_reference(inst, grid_res=24, time_limit=180.0)
        assert cip.covered_value_norm >= grid.covered_value_norm - 1e-6, (
            f"[seed {seed}] CIP reference {cip.covered_value_norm:.4f} < grid reference "
            f"{grid.covered_value_norm:.4f}; the exact candidate set lost coverage a "
            "24x24 grid found"
        )


def finer_grid_never_hurts():
    """More candidate sites can only relax the grid restriction."""
    inst = _toy()
    coarse = mclp_reference(inst, grid_res=8, time_limit=120.0)
    fine = mclp_reference(inst, grid_res=16, time_limit=120.0)
    assert fine.covered_value_norm >= coarse.covered_value_norm - 1e-6, (
        f"finer grid ({fine.covered_value_norm:.4f}) scored below coarser "
        f"({coarse.covered_value_norm:.4f}); the grid bound must be monotone"
    )
    assert fine.n_sites > coarse.n_sites


def coverage_uses_3d_distance():
    """A client beyond the 3-D radius but inside the 2-D one must NOT count.

    Regression guard for the 2-D/3-D mismatch: with R barely above the site
    altitude, a 2-D test covers far more clients than a 3-D one.
    """
    z_mid = 0.5 * (AREA["z"][0] + AREA["z"][1])
    inst = _toy(R=z_mid * 1.05)  # radius only just exceeds the altitude
    ref = mclp_reference(inst, grid_res=10, time_limit=120.0)
    # Under a 2-D test the altitude is free and many clients fall in range;
    # under 3-D almost none can, since sqrt(dz^2 + dxy^2) <= R leaves a tiny
    # ground footprint.
    assert ref.n_covered < inst.N, (
        "every client covered at a radius barely above the site altitude — "
        "coverage is ignoring the altitude term (2-D regression)"
    )


def normalisation_uses_the_solved_set():
    """covered_value_norm must be a fraction of what was actually solved."""
    inst = _toy(N=40)
    ref = mclp_reference(inst, grid_res=10, time_limit=120.0, max_clients=20)
    assert 0.0 <= ref.covered_value_norm <= 1.0 + 1e-9, (
        f"norm {ref.covered_value_norm:.4f} outside [0,1] — subsampled covered "
        "value is being divided by the full-instance total"
    )


def gap_is_reported():
    inst = _toy()
    ref = mclp_reference(inst, grid_res=10, time_limit=120.0)
    assert hasattr(ref, "mip_gap"), "MCLPResult must carry the residual MIP gap"
    if ref.optimal:
        assert np.isnan(ref.mip_gap) or ref.mip_gap < 1e-2, (
            f"claimed optimal with gap {ref.mip_gap}"
        )


if __name__ == "__main__":
    check("optimum dominates a random heuristic", optimum_dominates_a_heuristic)
    check("CIP optimum dominates the real optimizers", cip_optimum_dominates_the_actual_optimizers)
    check("CIP reference is at least the grid reference", cip_reference_is_at_least_the_grid_reference)
    check("finer grid never scores lower", finer_grid_never_hurts)
    check("coverage uses 3-D slant distance", coverage_uses_3d_distance)
    check("normalisation uses the solved set", normalisation_uses_the_solved_set)
    check("MIP gap is reported", gap_is_reported)
    finish()
