"""Al-Hourani path-loss model: monotonicity, bounds, inverse consistency."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from _lib import check, finish

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


def los_monotone():
    a, b = SUBURBAN["a"], SUBURBAN["b"]
    ps = [los_probability(e, a, b) for e in (5, 15, 30, 60, 90)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(p2 > p1 for p1, p2 in zip(ps, ps[1:]))


def path_loss_monotone_in_distance():
    for env in ENV_PRESETS.values():
        pls = [average_path_loss(r, 100.0, FREQ_HZ, **env) for r in (0, 100, 500, 1000, 5000)]
        assert all(p2 > p1 for p1, p2 in zip(pls, pls[1:]))


def radius_at_boundary_loss():
    r = coverage_radius(100.0, MAX_PL_DB, FREQ_HZ, **SUBURBAN)
    assert r > 0.0
    assert average_path_loss(r, 100.0, FREQ_HZ, **SUBURBAN) <= MAX_PL_DB + 1e-6
    assert average_path_loss(r + 1.0, 100.0, FREQ_HZ, **SUBURBAN) > MAX_PL_DB
    assert coverage_radius(100.0, 30.0, FREQ_HZ, **SUBURBAN) == 0.0  # infeasible budget


def mozaffari_altitude_interior_peak():
    h_star, r_star = optimal_altitude_mozaffari(
        MAX_PL_DB, FREQ_HZ, **SUBURBAN, h_min_m=20.0, h_max_m=2000.0
    )
    assert 20.0 <= h_star <= 2000.0 and r_star > 0.0
    r_lo = coverage_radius(20.0, MAX_PL_DB, FREQ_HZ, **SUBURBAN)
    r_hi = coverage_radius(2000.0, MAX_PL_DB, FREQ_HZ, **SUBURBAN)
    assert r_star >= max(r_lo, r_hi)  # optimum beats both band edges


def min_altitude_inverse():
    h, r = min_altitude_for_radius(800.0, MAX_PL_DB, FREQ_HZ, **SUBURBAN, h_min_m=20.0, h_max_m=2000.0)
    assert 20.0 <= h <= 2000.0 and r >= 800.0
    assert coverage_radius(h, MAX_PL_DB, FREQ_HZ, **SUBURBAN) >= 800.0 - 1e-6
    # Unreachable requirement falls back to best achievable, placement proceeds.
    h2, r2 = min_altitude_for_radius(1e9, MAX_PL_DB, FREQ_HZ, **SUBURBAN, h_min_m=20.0, h_max_m=2000.0)
    assert 20.0 <= h2 <= 2000.0 and 0.0 < r2 < 1e9


def presets_finite():
    for name, env in ENV_PRESETS.items():
        r = coverage_radius(120.0, 110.0, FREQ_HZ, **env)
        assert np.isfinite(r) and r >= 0.0, name


check("P(LoS) in [0,1] and increasing with elevation", los_monotone)
check("path loss strictly increases with ground distance (all presets)", path_loss_monotone_in_distance)
check("coverage radius sits exactly at the loss budget; infeasible -> 0", radius_at_boundary_loss)
check("Mozaffari optimal altitude is an interior peak", mozaffari_altitude_interior_peak)
check("min_altitude_for_radius inverse-consistent + graceful fallback", min_altitude_inverse)
check("every ENV preset yields a finite radius", presets_finite)
finish()
