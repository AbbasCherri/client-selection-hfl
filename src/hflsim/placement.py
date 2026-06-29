"""PSO integration bridge: run placement optimization on real HFL client coordinates.

Converts the {client_id: (lat, lon)} dict produced by data_loader into a
ProblemInstance in projected metric space, runs the requested optimizer, and
returns a list of UAVAggregator objects at the optimized positions.

This is the Tier-2 integration point described in the uav-pso-bench README.
"""

from __future__ import annotations

import math

import numpy as np

from uavbench.optimizers import REGISTRY
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import ProblemInstance
from hflsim.shared.coords import latlon_to_meters
from hflsim.simulation.uav import UAVAggregator

_EARTH_RADIUS_M = 6_371_000.0


def pso_place_uavs(
    client_coords: dict[int, tuple[float, float]],
    K: int,
    R_comm: float = 50_000.0,
    capacity: int = 20,
    seed: int = 42,
    method: str = "pso",
    P: int = 50,
    G_max: int = 100,
) -> list[UAVAggregator]:
    """Run a placement optimizer on real client coordinates and return UAVAggregators.

    Parameters
    ----------
    client_coords:
        Mapping of client_id → (lat, lon) in degrees, as returned by
        ``get_hfl_data_partitions()``.
    K:
        Number of UAVs to place (matches the ``--U`` CLI argument).
    R_comm:
        UAV–IoT communication range in metres (default 50 km covers Noto Peninsula).
    capacity:
        Max clients per UAV (passed through to UAVAggregator).
    seed:
        Random seed for both instance generation and optimizer.
    method:
        Optimizer key from ``uavbench.optimizers.REGISTRY``
        (``"pso"``, ``"ga"``, ``"centroid"``, ``"random"``, ``"static"``).
    P:
        Population size (ignored for heuristic methods).
    G_max:
        Maximum generations / iterations (ignored for heuristic methods).
    """
    latlon = np.array(list(client_coords.values()), dtype=np.float64)  # (N, 2)
    xy_m, ref = latlon_to_meters(latlon)  # projected metres, ref = (lat0, lon0)
    N = len(xy_m)

    device_coords = np.column_stack([xy_m, np.zeros(N)])

    lower = np.array([xy_m[:, 0].min(), xy_m[:, 1].min(), 50.0])
    upper = np.array([xy_m[:, 0].max(), xy_m[:, 1].max(), 200.0])

    # Start with no previous layout (all zeros → movement penalty is from origin).
    prev_positions = np.zeros((K, 3))

    instance = ProblemInstance(
        device_coords=device_coords,
        value=np.ones(N),  # uniform weights for placement-only pass
        capacity=np.full(K, float(capacity)),
        battery=np.ones(K),
        prev_positions=prev_positions,
        lower=lower,
        upper=upper,
        R_comm=R_comm,
        B_min_uav=0.0,  # battery is always 1.0 here, gate is always satisfied
    )

    fitness = Fitness(instance)

    rng = np.random.default_rng(seed)

    cls = REGISTRY[method]
    if method in ("pso", "ga"):
        optimizer = cls(P=P, G_max=G_max)
    else:
        optimizer = cls()

    result = optimizer.optimize(instance, fitness, rng)

    # Convert optimised metre positions back to lat/lon for UAVAggregator.
    positions = result.best_position.reshape(K, 3)
    lat0_rad = math.radians(ref[0])
    uavs = []
    for i, (x, y, _z) in enumerate(positions):
        lat = ref[0] + math.degrees(y / _EARTH_RADIUS_M)
        lon = ref[1] + math.degrees(x / (_EARTH_RADIUS_M * math.cos(lat0_rad)))
        uavs.append(UAVAggregator(uav_id=i, coords=(lat, lon), capacity=capacity))
    return uavs
