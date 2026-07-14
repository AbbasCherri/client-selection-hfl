"""Path-loss model invariants: monotonicity, bounds, and inverse consistency."""

import numpy as np

from uavbench.problem.path_loss import (
    ENV_PRESETS,
    average_path_loss,
    coverage_radius,
    los_probability,
    min_altitude_for_radius,
    optimal_altitude_mozaffari,
)

FREQ_HZ = 2e9
MAX_PL_DB = 100.0
SUBURBAN = ENV_PRESETS["suburban"]


def test_los_probability_increases_with_elevation():
    a, b = SUBURBAN["a"], SUBURBAN["b"]
    ps = [los_probability(e, a, b) for e in (5, 15, 30, 60, 90)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(p2 > p1 for p1, p2 in zip(ps, ps[1:]))


def test_path_loss_monotonic_in_ground_distance():
    # At fixed altitude, moving outward both lengthens the slant path and
    # lowers P(LoS) — loss must strictly increase (the bisection's premise).
    for env in ENV_PRESETS.values():
        pls = [average_path_loss(r, 100.0, FREQ_HZ, **env) for r in (0, 100, 500, 1000, 5000)]
        assert all(p2 > p1 for p1, p2 in zip(pls, pls[1:]))


def test_coverage_radius_at_boundary_loss():
    r = coverage_radius(100.0, MAX_PL_DB, FREQ_HZ, **SUBURBAN)
    assert r > 0.0
    # The found radius sits within budget; a point just beyond exceeds it.
    assert average_path_loss(r, 100.0, FREQ_HZ, **SUBURBAN) <= MAX_PL_DB + 1e-6
    assert average_path_loss(r + 1.0, 100.0, FREQ_HZ, **SUBURBAN) > MAX_PL_DB


def test_coverage_radius_zero_when_budget_infeasible():
    assert coverage_radius(100.0, 30.0, FREQ_HZ, **SUBURBAN) == 0.0


def test_optimal_altitude_within_bounds_and_unimodal_peak():
    h_star, r_star = optimal_altitude_mozaffari(
        MAX_PL_DB, FREQ_HZ, **SUBURBAN, h_min_m=20.0, h_max_m=2000.0
    )
    assert 20.0 <= h_star <= 2000.0
    assert r_star > 0.0
    # The optimum beats both band edges (interior peak of the unimodal curve).
    r_lo = coverage_radius(20.0, MAX_PL_DB, FREQ_HZ, **SUBURBAN)
    r_hi = coverage_radius(2000.0, MAX_PL_DB, FREQ_HZ, **SUBURBAN)
    assert r_star >= max(r_lo, r_hi)


def test_min_altitude_inverse_consistency():
    required = 800.0
    h, r = min_altitude_for_radius(
        required, MAX_PL_DB, FREQ_HZ, **SUBURBAN, h_min_m=20.0, h_max_m=2000.0
    )
    assert 20.0 <= h <= 2000.0
    assert r >= required
    # And the standalone radius at that altitude confirms the requirement.
    assert coverage_radius(h, MAX_PL_DB, FREQ_HZ, **SUBURBAN) >= required - 1e-6


def test_min_altitude_fallback_when_unreachable():
    # An absurd requirement can't be met: fall back to the best achievable
    # radius rather than failing, so placement always proceeds.
    h, r = min_altitude_for_radius(
        1e9, MAX_PL_DB, FREQ_HZ, **SUBURBAN, h_min_m=20.0, h_max_m=2000.0
    )
    assert 20.0 <= h <= 2000.0
    assert 0.0 < r < 1e9


def test_all_env_presets_produce_finite_radii():
    for name, env in ENV_PRESETS.items():
        r = coverage_radius(120.0, 110.0, FREQ_HZ, **env)
        assert np.isfinite(r) and r >= 0.0, name
