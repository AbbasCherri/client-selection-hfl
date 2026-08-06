"""Air-to-ground path-loss model and altitude-coupled coverage radius.

Implements the probabilistic LoS/NLoS air-to-ground channel from **two**
distinct Al-Hourani et al. papers, which do different jobs and are cited
separately (not as one merged "2014" source):

* Al-Hourani, Kandeepan & Jamalipour, "Modeling Air-to-Ground Path Loss for
  Low Altitude Platforms in Urban Environments," IEEE GLOBECOM 2014 — the
  source of the environment shape constants and, critically, the
  `eta_los_db`/`eta_nlos_db` excess-loss values (that paper's `mu_1`/`mu_2`,
  Table II, 2000 MHz row).
* Al-Hourani, Kandeepan & Lardner, "Optimal LAP Altitude for Maximum
  Coverage," IEEE WCL 2014 — the source of the S-curve LoS-probability fit
  (`los_probability`), the 3-D free-space path-loss form, and the
  altitude/coverage-radius optimization this module implements.

This channel model is what both placement literature baselines build on:

* Mozaffari, Saad, Bennis & Debbah, "Efficient Deployment of Multiple
  Unmanned Aerial Vehicles for Optimal Wireless Coverage," IEEE Comm.
  Letters 2016 (coverage radius maximized over altitude).
* Alzenad, El-Keyi, Lagum & Yanikomeroglu, "3-D Placement of an Unmanned
  Aerial Vehicle Base Station (UAV-BS) for Energy-Efficient Maximal
  Coverage," IEEE WCL 2017 (minimum altitude achieving a required radius).

**Sourcing status (resolved 2026-07):**
- `eta_los_db`/`eta_nlos_db` in `ENV_PRESETS` are confirmed to match the
  GLOBECOM paper's Table II `(mu_1, mu_2)` at 2000 MHz exactly (Suburban
  0.1/21, Urban 1.0/20, Dense Urban 1.6/23, Highrise 2.3/34 dB).
- `average_path_loss`'s free-space term is confirmed evaluated at the 3-D
  slant distance (`d3d = hypot(distance_ground_m, altitude_m)`), matching
  the WCL letter's own `d = sqrt(h^2 + r^2)` convention exactly — not the
  horizontal-only distance.
- The `(a, b)` LoS shape constants in `ENV_PRESETS` are the standard
  literature-cited Al-Hourani values, used as-is rather than re-derived
  from the WCL letter's bivariate surface-fit polynomial over `(alpha*beta,
  gamma)` — a stated simplification, not an unverified gap.
- Each baseline's own optimum-derivation fidelity (Mozaffari's closed-form
  altitude vs. this module's grid search; Alzenad's smallest-enclosing-
  circle vs. centroid) is *not* covered by the two papers above and is
  documented as an explicit adaptation choice in `optimizers/mozaffari2016.py`
  / `optimizers/alzenad2017.py` respectively — see those files, not here.

Units are stated per parameter: metres (m), Hertz (Hz), decibels (dB),
degrees (deg).
"""

from __future__ import annotations

import math

import numpy as np

_C = 299_792_458.0  # speed of light, m/s

# Al-Hourani et al. environment presets: a/b shape the sigmoid LoS
# probability (standard literature-cited values); eta_los/eta_nlos are mean
# excess losses (dB) beyond free space for LoS/NLoS links, confirmed against
# the GLOBECOM 2014 paper's Table II (mu_1, mu_2 at 2000 MHz).
ENV_PRESETS: dict[str, dict[str, float]] = {
    "suburban": {"a": 4.88, "b": 0.43, "eta_los_db": 0.1, "eta_nlos_db": 21.0},
    "urban": {"a": 9.61, "b": 0.16, "eta_los_db": 1.0, "eta_nlos_db": 20.0},
    # b = 0.11 and NOT the 0.114 printed in Moon et al. 2022 (Electronics
    # 11(7):1036, Table 3). Resolved by cross-check, not by preference: the
    # radius-maximizing elevation angle is a pure function of (a, b), and every
    # source — including Moon et al. themselves — reports 54.62 deg for dense
    # urban. b = 0.11 gives 54.619; b = 0.114 gives 53.83. Their table and their
    # own stated optimum are inconsistent, and the optimum is the checkable one.
    # Pinned by check_mclp.py::optimal_elevation_matches_the_published_values.
    "dense_urban": {"a": 12.08, "b": 0.11, "eta_los_db": 1.6, "eta_nlos_db": 23.0},
    "high_rise_urban": {"a": 27.23, "b": 0.08, "eta_los_db": 2.3, "eta_nlos_db": 34.0},
}


def los_probability(elevation_deg: float, a: float, b: float) -> float:
    """Probability of a line-of-sight link at the given elevation angle.

    ``P(LoS) = 1 / (1 + a * exp(-b * (elevation_deg - a)))`` — the standard
    sigmoid of Al-Hourani et al.

    Parameters
    ----------
    elevation_deg : ground-to-UAV elevation angle (deg), in [0, 90].
    a, b          : environment shape constants (dimensionless).
    """
    return 1.0 / (1.0 + a * math.exp(-b * (elevation_deg - a)))


def average_path_loss(
    distance_ground_m: float,
    altitude_m: float,
    freq_hz: float,
    a: float,
    b: float,
    eta_los_db: float,
    eta_nlos_db: float,
) -> float:
    """LoS-probability-weighted mean path loss (dB) for one ground link.

    Free-space loss at the 3D slant distance plus the P(LoS)-weighted mean
    of the LoS/NLoS excess attenuations:

        PL = FSPL(d_3d, f) + P_los * eta_los + (1 - P_los) * eta_nlos

    Parameters
    ----------
    distance_ground_m : horizontal ground distance UAV-to-device (m), >= 0.
    altitude_m        : UAV altitude above ground (m), > 0.
    freq_hz           : carrier frequency (Hz).
    a, b, eta_los_db, eta_nlos_db : environment constants (see ENV_PRESETS).
    """
    d3d = math.hypot(distance_ground_m, altitude_m)
    elevation_deg = math.degrees(math.atan2(altitude_m, max(distance_ground_m, 1e-9)))
    p_los = los_probability(elevation_deg, a, b)
    fspl_db = (
        20.0 * math.log10(max(d3d, 1e-9))
        + 20.0 * math.log10(freq_hz)
        + 20.0 * math.log10(4.0 * math.pi / _C)
    )
    return fspl_db + p_los * eta_los_db + (1.0 - p_los) * eta_nlos_db


def coverage_radius(
    altitude_m: float,
    max_path_loss_db: float,
    freq_hz: float,
    a: float,
    b: float,
    eta_los_db: float,
    eta_nlos_db: float,
    r_search_max_m: float = 20_000.0,
) -> float:
    """Largest ground radius (m) whose mean path loss stays within budget.

    Bisection over horizontal distance for the largest ``r`` with
    ``average_path_loss(r, altitude, ...) <= max_path_loss_db``. Assumes
    path loss increases monotonically in ``r`` at fixed altitude (holds for
    these presets since both slant distance and NLoS probability grow with
    ``r``); returns 0.0 when even the nadir link exceeds the budget.

    Parameters
    ----------
    altitude_m       : UAV altitude (m), > 0.
    max_path_loss_db : maximum tolerable mean path loss (dB).
    freq_hz          : carrier frequency (Hz).
    r_search_max_m   : upper bracket for the bisection search (m).
    """
    args = (freq_hz, a, b, eta_los_db, eta_nlos_db)
    if average_path_loss(0.0, altitude_m, *args) > max_path_loss_db:
        return 0.0
    if average_path_loss(r_search_max_m, altitude_m, *args) <= max_path_loss_db:
        return r_search_max_m
    lo, hi = 0.0, r_search_max_m
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if average_path_loss(mid, altitude_m, *args) <= max_path_loss_db:
            lo = mid
        else:
            hi = mid
    return lo


def optimal_elevation_angle(
    a: float,
    b: float,
    eta_los_db: float,
    eta_nlos_db: float,
) -> float:
    """Elevation angle (degrees) maximizing the coverage radius — Al-Hourani Eq. (7).

        (pi / (9 ln10)) * tan(theta)
            + a*b*A*exp(-b(180/pi * theta - a)) / (a*exp(-b(180/pi * theta - a)) + 1)^2 = 0

    with ``A = eta_los - eta_nlos`` and ``theta`` in radians. The striking
    property, and the reason this is worth having as its own function: the
    optimum depends **only on the environment** — not on the loss budget, the
    frequency, or the altitude band. Alzenad et al. (2017) report 20.34 deg
    (suburban), 42.44 (urban), 54.62 (dense urban) and 75.52 (high-rise); those
    values are asserted against this solver in
    ``tests/sanity_checks/check_mclp.py``.

    Solved by bisection: the left-hand side rises monotonically from negative
    (where the LoS term dominates) to +inf as theta -> 90 deg.
    """
    big_a = eta_los_db - eta_nlos_db

    def lhs(theta_rad: float) -> float:
        deg = math.degrees(theta_rad)
        e = math.exp(-b * (deg - a))
        return (
            math.pi / (9.0 * math.log(10.0)) * math.tan(theta_rad)
            + a * b * big_a * e / (a * e + 1.0) ** 2
        )

    lo, hi = 1e-6, math.radians(89.999)
    if lhs(lo) > 0.0:
        return math.degrees(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lhs(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi))


def optimal_altitude_mozaffari(
    max_path_loss_db: float,
    freq_hz: float,
    a: float,
    b: float,
    eta_los_db: float,
    eta_nlos_db: float,
    h_min_m: float = 20.0,
    h_max_m: float = 2000.0,
    n_grid: int = 200,
) -> tuple[float, float]:
    """Altitude maximizing the coverage radius, per Mozaffari et al. 2016.

    Grid search over ``[h_min_m, h_max_m]`` for the ``h`` maximizing
    ``coverage_radius(h, ...)``. The radius-vs-altitude curve is unimodal
    under this model (rising while improved LoS dominates, falling once
    free-space loss dominates), so a grid of ``n_grid`` points suffices.
    Known deviation (not verified against the source derivation): Mozaffari
    2016 (Eq. 10-12) derives this optimum in closed form; this grid search
    is a numerical stand-in, adopted deliberately rather than re-deriving
    the closed form. See `optimizers/mozaffari2016.py` for the fidelity
    note this feeds.

    Returns ``(h_star_m, r_star_m)``.
    """
    hs = np.linspace(h_min_m, h_max_m, n_grid)
    radii = [
        coverage_radius(float(h), max_path_loss_db, freq_hz, a, b, eta_los_db, eta_nlos_db)
        for h in hs
    ]
    i = int(np.argmax(radii))
    return float(hs[i]), float(radii[i])


def min_altitude_for_radius(
    required_radius_m: float,
    max_path_loss_db: float,
    freq_hz: float,
    a: float,
    b: float,
    eta_los_db: float,
    eta_nlos_db: float,
    h_min_m: float = 20.0,
    h_max_m: float = 2000.0,
    n_grid: int = 200,
) -> tuple[float, float]:
    """Lowest altitude whose coverage radius reaches the requirement.

    Alzenad et al. 2017's decoupled altitude step: among altitudes in
    ``[h_min_m, h_max_m]``, pick the smallest ``h`` with
    ``coverage_radius(h, ...) >= required_radius_m`` (energy-efficient:
    minimum transmit-power altitude that still covers the target circle).
    If no altitude achieves the requirement, fall back to the altitude
    maximizing the radius, so callers always get a usable placement.
    Known deviation (not verified against the source derivation): Alzenad
    2017 (§III) frames this via a circle-placement + feasibility argument;
    this scan is a numerical stand-in for the same minimum-feasible-
    altitude choice, adopted deliberately. See `optimizers/alzenad2017.py`
    for the fidelity note this feeds.

    Returns ``(h_m, r_achieved_m)``.
    """
    hs = np.linspace(h_min_m, h_max_m, n_grid)
    best_h, best_r = float(hs[0]), 0.0
    for h in hs:
        r = coverage_radius(float(h), max_path_loss_db, freq_hz, a, b, eta_los_db, eta_nlos_db)
        if r >= required_radius_m:
            return float(h), float(r)
        if r > best_r:
            best_h, best_r = float(h), float(r)
    return best_h, best_r


def path_loss_db_for_radius(
    target_radius_m: float,
    freq_hz: float,
    a: float,
    b: float,
    eta_los_db: float,
    eta_nlos_db: float,
    h_min_m: float = 20.0,
    h_max_m: float = 120.0,
    lo_db: float = 60.0,
    hi_db: float = 200.0,
    tol_m: float = 1.0,
) -> float:
    """Max-path-loss budget whose derived coverage radius ≈ ``target_radius_m``.

    Inverts :func:`optimal_altitude_mozaffari` (monotone in the link budget) by
    bisection. This is the equal-radius calibration knob: the path-loss
    placement baselines derive their own coverage radius from the link budget,
    so comparing them against a system gated at ``R_comm`` is only fair if their
    derived radius matches ``R_comm``. Without it, a baseline tuned for a 20 km
    radius but gated at 2 km spreads its UAVs far too thin and loses on the
    radius mismatch rather than on its placement rule.
    """
    for _ in range(60):
        mid = 0.5 * (lo_db + hi_db)
        _h, r = optimal_altitude_mozaffari(
            mid, freq_hz, a, b, eta_los_db, eta_nlos_db, h_min_m=h_min_m, h_max_m=h_max_m
        )
        if abs(r - target_radius_m) <= tol_m:
            return float(mid)
        if r < target_radius_m:
            lo_db = mid
        else:
            hi_db = mid
    return float(0.5 * (lo_db + hi_db))
