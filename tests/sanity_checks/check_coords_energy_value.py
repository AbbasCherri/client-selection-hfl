"""Geo projection, energy model, and V_i value-function checks (real-data path)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from _lib import check, finish

from hflsim.shared.coords import haversine, haversine_matrix, latlon_to_meters
from hflsim.shared.value import beta_schedule, compute_utility, compute_value
from uavbench.problem.energy import EnergyModel


def haversine_known_values():
    assert abs(haversine((37.488, 137.272), (37.488, 137.272))) < 1e-6
    assert abs(haversine((37.0, 137.0), (38.0, 138.0)) - haversine((38.0, 138.0), (37.0, 137.0))) < 1e-6
    d = haversine((0.0, 0.0), (0.0, 1.0))
    assert 111_000 < d < 111_500  # 1 deg longitude at the equator
    d = haversine((0.0, 0.0), (0.0, 180.0))
    assert 19_900_000 < d < 20_100_000  # half circumference


def projection_scales_and_roundtrip():
    xy, ref = latlon_to_meters(np.array([[37.488, 137.272]]))
    assert abs(xy[0, 0]) < 1e-6 and abs(xy[0, 1]) < 1e-6
    xy, _ = latlon_to_meters(np.array([[37.0, 137.0], [38.0, 137.0]]))
    assert 110_000 < abs(xy[1, 1] - xy[0, 1]) < 112_000  # ~111 km/deg latitude
    xy, _ = latlon_to_meters(np.array([[37.0, 137.0], [37.0, 138.0]]))
    assert 85_000 < abs(xy[1, 0] - xy[0, 0]) < 92_000  # cos(37 deg) shrink
    # Euclidean-after-projection agrees with haversine for short hops.
    a, b = (37.488, 137.272), (37.510, 137.290)
    xy, _ = latlon_to_meters(np.array([a, b]))
    eucl = float(np.hypot(*(xy[0] - xy[1])))
    assert abs(eucl - haversine(a, b)) / haversine(a, b) < 0.001


def matrix_matches_scalar():
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.uniform(36.5, 38.0, 12), rng.uniform(136.5, 138.0, 12)])
    b = np.column_stack([rng.uniform(36.5, 38.0, 5), rng.uniform(136.5, 138.0, 5)])
    mat = haversine_matrix(a, b)
    assert mat.shape == (12, 5)
    for i in (0, 5, 11):
        for j in (0, 4):
            assert abs(mat[i, j] - haversine(tuple(a[i]), tuple(b[j]))) < 1e-6


def energy_hand_computed():
    m = EnergyModel()
    assert m.energy_joules(0.0) == m.p_hover * m.t_serve  # pure hover
    e0, e1, e2 = m.energy_joules(0.0), m.energy_joules(150.0), m.energy_joules(300.0)
    assert abs((e2 - e1) - (e1 - e0)) < 1e-9  # linear in distance
    assert abs((e1 - e0) - m.p_fly * 150.0 / m.cruise_speed) < 1e-9
    # Defaults: p_fly=250 W, p_hover=200 W, cruise=15 m/s, t_serve=60 s.
    assert abs(m.energy_joules(1500.0) - (250.0 * 100.0 + 200.0 * 60.0)) < 1e-6
    d = 4321.0
    assert abs(m.battery_fraction(d) - m.energy_joules(d) / m.battery_capacity_j) < 1e-12


def beta_schedule_endpoints():
    assert abs(beta_schedule(0) - 1.0) < 1e-9
    assert abs(beta_schedule(20, T_decay=20)) < 1e-9
    assert abs(beta_schedule(1000)) < 1e-9  # clamped past T_decay
    assert abs(beta_schedule(10, T_decay=20) - 0.5) < 1e-9
    betas = [beta_schedule(t) for t in range(25)]
    assert all(b1 >= b2 for b1, b2 in zip(betas, betas[1:]))


def value_bounds_and_modes():
    rng = np.random.default_rng(0)
    N, K = 5, 2
    dc = np.column_stack([rng.uniform(0, 1000, N), rng.uniform(0, 1000, N), np.zeros(N)])
    epi = np.array([500.0, 500.0, 0.0])
    snr = rng.uniform(0.0, 30.0, N)
    samp = rng.integers(20, 200, N).astype(float)
    prev = rng.uniform(0, 1000, (K, 3))
    rep = rng.beta(2, 2, N)
    u = compute_utility(dc, epi, snr, samp, prev)
    assert u.shape == (N,) and np.all((0.0 <= u) & (u <= 1.0))
    # pinned beta=1 -> V == U; scheduled at t=T_decay -> V == reputation.
    v_pin = compute_value(dc, epi, snr, samp, prev, rep, beta_mode="pinned")
    assert np.allclose(v_pin, u)
    v_late = compute_value(dc, epi, snr, samp, prev, rep, beta_mode="scheduled", t=20, T_decay=20)
    assert np.allclose(v_late, rep)
    try:
        compute_value(dc, epi, snr, samp, prev, rep, beta_mode="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid beta_mode must raise")


check("haversine known values (equator/antipodal/symmetry)", haversine_known_values)
check("latlon->metres scales and haversine round-trip", projection_scales_and_roundtrip)
check("vectorized haversine_matrix matches scalar haversine", matrix_matches_scalar)
check("energy model matches hand-computed values", energy_hand_computed)
check("beta schedule endpoints and monotone decay", beta_schedule_endpoints)
check("V_i in [0,1]; pinned/scheduled modes; invalid mode raises", value_bounds_and_modes)
finish()
