"""Altitude-dependent coverage radius — the link model the objective is gated on.

Why this replaces the hard range gate
-------------------------------------
The benchmark's original coverage test was ``slant_distance <= R_comm`` with
``R_comm`` a constant. That is not a channel: it says a UAV at 20 m and one at
120 m have the same link budget, so climbing only ever spends radius on altitude
and the ground disc ``sqrt(R_comm^2 - z^2)`` shrinks monotonically. Under such a
gate a correct 3D optimizer will always drive every UAV to the altitude floor —
the vertical dimension is degenerate, and "3D placement" reduces to a planar
placement carrying a height column.

The real air-to-ground channel does not behave that way. Climbing raises the
elevation angle, which raises P(LoS) and removes the large NLoS excess loss
(21 dB in suburban terrain); climbing further eventually loses to free-space
loss. The resulting radius-versus-altitude curve is **unimodal with an interior
maximum** — which is precisely why the UAV placement literature treats altitude
as a decision variable at all. :mod:`.path_loss` already implements that channel
(Al-Hourani et al. 2014); this module makes it the gate every method is scored
on, rather than something only the two path-loss baselines consulted.

Calibration: ``R_comm`` keeps its meaning
-----------------------------------------
Swapping in a channel would otherwise make every configured ``R_comm`` value
meaningless. Instead the link budget is *calibrated per instance* so that the
**best achievable** ground radius, maximized over altitude, equals ``R_comm``:

    PL* = min_z  average_path_loss(R_comm, z)

At the minimizing altitude ``z*`` the radius is exactly ``R_comm``; at every
other altitude it is strictly less. So a configured ``R_comm = 500 m`` still
means "500 m of reach", now qualified by "if you fly at the right height" — and
every existing sweep value keeps its interpretation.

This also dissolves the equal-radius problem at its root. The 618-vs-500 m
handicap existed because mozaffari2016 derived its own radius from its own
altitude while everyone else was gated at a flat ``R_comm``. Here every method's
radius comes from its own altitude through one shared model, so there is no
private ruler left to hold.
"""

from __future__ import annotations

import numpy as np

from .path_loss import ENV_PRESETS, average_path_loss, coverage_radius

# LUT construction runs a 60-step bisection per altitude level; a placement run
# builds many Fitness objects over the same few (R_comm, band, environment)
# combinations, so the table is built once and shared.
_LUT_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, float, float]] = {}


class LinkModel:
    """Maps UAV altitude to the ground radius it can serve.

    Vectorized through a precomputed lookup table with linear interpolation:
    the exact radius needs a bisection per altitude, which is far too slow to
    call inside a fitness evaluation that scores hundreds of candidates.
    """

    def __init__(
        self,
        r_comm_m: float,
        z_min_m: float,
        z_max_m: float,
        environment: str = "suburban",
        freq_ghz: float = 2.0,
        n_levels: int = 129,
    ) -> None:
        self.r_comm_m = float(r_comm_m)
        self.z_min_m, self.z_max_m = float(z_min_m), float(z_max_m)
        self.environment = environment
        self.freq_ghz = float(freq_ghz)
        key = (self.r_comm_m, self.z_min_m, self.z_max_m, environment, self.freq_ghz, n_levels)
        if key not in _LUT_CACHE:
            _LUT_CACHE[key] = self._build(n_levels)
        self._z_grid, self._r_grid, self.max_path_loss_db, self.z_star_m = _LUT_CACHE[key]

    def _build(self, n_levels: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        env = ENV_PRESETS[self.environment]
        freq_hz = self.freq_ghz * 1e9
        z_grid = np.linspace(self.z_min_m, self.z_max_m, n_levels)

        # Budget that makes R_comm reachable from the single best altitude in the
        # band. Searching the *band* rather than an unbounded range matters: if
        # the unconstrained optimum lies above z_max, calibrating to it would put
        # R_comm out of reach everywhere the UAV is actually allowed to fly.
        losses = np.array(
            [average_path_loss(self.r_comm_m, float(z), freq_hz, **env) for z in z_grid]
        )
        i = int(np.argmin(losses))
        pl_star = float(losses[i])
        z_star = float(z_grid[i])

        r_grid = np.array(
            [coverage_radius(float(z), pl_star, freq_hz, **env) for z in z_grid]
        )
        return z_grid, r_grid, pl_star, z_star

    def radius(self, z: np.ndarray | float) -> np.ndarray:
        """Ground radius (m) served from altitude ``z``; shape follows the input.

        Altitudes outside the calibrated band clamp to its endpoints rather than
        extrapolating — the search box bounds z anyway, and a linear
        extrapolation of a unimodal curve would invent radius past the edge.
        """
        return np.interp(np.asarray(z, dtype=np.float64), self._z_grid, self._r_grid)

    def slant_radius(self, z: np.ndarray | float) -> np.ndarray:
        """Slant-distance gate equivalent to the ground radius at altitude ``z``.

        :meth:`radius` is a **ground** radius (the path-loss model is written in
        horizontal distance), but the assignment compares the **3D slant**
        device-to-UAV distance. Comparing a slant distance against a ground
        radius would silently under-cover by the altitude term. Since
        ``slant = sqrt(ground^2 + z^2)`` is monotone in ``ground`` at fixed
        ``z``, gating slant against ``sqrt(r_ground^2 + z^2)`` is exactly
        equivalent to gating ground against ``r_ground``.
        """
        zz = np.asarray(z, dtype=np.float64)
        r = self.radius(zz)
        return np.sqrt(r * r + zz * zz)

    def min_altitude_for_radius(self, required_r_m: float) -> float:
        """Lowest in-band altitude whose ground radius reaches ``required_r_m``.

        Alzenad et al.'s rule, expressed against the shared model. Falls back to
        ``z_star_m`` when no altitude in the band reaches the requirement, which
        is the best available rather than an infeasibility — the shortfall then
        shows up honestly as uncovered devices.
        """
        ok = self._r_grid >= required_r_m
        if not ok.any():
            return self.z_star_m
        return float(self._z_grid[int(np.argmax(ok))])

    @property
    def r_max_m(self) -> float:
        """Best achievable radius over the band — equals ``r_comm_m`` by construction."""
        return float(self._r_grid.max())

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"LinkModel(r_comm={self.r_comm_m:.0f}m, z*={self.z_star_m:.0f}m, "
            f"PL*={self.max_path_loss_db:.1f}dB, env={self.environment})"
        )
