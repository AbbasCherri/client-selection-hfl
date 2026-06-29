"""Shared coordinate utilities used by both hflsim and uavbench.

haversine()       — great-circle distance in metres between two (lat, lon) pairs
latlon_to_meters() — equirectangular projection of (lat, lon) arrays to local (x, y) metres
"""

from __future__ import annotations

import math

import numpy as np

_EARTH_RADIUS_M = 6_371_000.0


def haversine(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """Return the Haversine (great-circle) distance in metres between two (lat, lon) pairs."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return _EARTH_RADIUS_M * c


def latlon_to_meters(
    coords_latlon: np.ndarray,
    ref: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Project (lat, lon) degrees to local (x, y) metres via equirectangular map.

    Parameters
    ----------
    coords_latlon:
        ``(N, 2)`` array of ``(lat, lon)`` in degrees.
    ref:
        Reference ``(lat, lon)`` for the projection origin. Defaults to the mean.

    Returns
    -------
    xy_meters, ref
        ``(N, 2)`` projected metres and the reference point used.
    """
    coords_latlon = np.asarray(coords_latlon, dtype=np.float64)
    if ref is None:
        ref = (float(coords_latlon[:, 0].mean()), float(coords_latlon[:, 1].mean()))
    lat0, lon0 = ref
    lat0_rad = np.radians(lat0)
    x = np.radians(coords_latlon[:, 1] - lon0) * np.cos(lat0_rad) * _EARTH_RADIUS_M
    y = np.radians(coords_latlon[:, 0] - lat0) * _EARTH_RADIUS_M
    return np.column_stack([x, y]), ref
