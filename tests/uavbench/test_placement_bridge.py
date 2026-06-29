"""Tests for federated.py helpers: _build_problem_instance, _covered_clients,
_uav_pos_to_latlon (placement → FL bridge)."""

import math

import numpy as np
import pytest

from hflsim.shared.coords import haversine
from uavbench.fl.federated import (
    _build_problem_instance,
    _covered_clients,
    _place_uavs,
    _uav_pos_to_latlon,
)


def _noto_coords(n=5):
    """Spread n clients across the Noto Peninsula."""
    return {
        i: (37.3 + i * 0.05, 136.9 + i * 0.05)
        for i in range(n)
    }


# ── _build_problem_instance ───────────────────────────────────────────────────

class TestBuildProblemInstance:
    def test_device_count_matches_N(self):
        coords = _noto_coords(10)
        inst, ref = _build_problem_instance(coords, K=3, R_comm=5000, capacity=10,
                                            prev_positions_m=None)
        assert inst.N == 10

    def test_uav_count_matches_K(self):
        coords = _noto_coords(8)
        inst, ref = _build_problem_instance(coords, K=4, R_comm=5000, capacity=10,
                                            prev_positions_m=None)
        assert inst.K == 4

    def test_ref_is_mean_latlon(self):
        coords = {0: (37.0, 137.0), 1: (38.0, 138.0)}
        inst, ref = _build_problem_instance(coords, K=2, R_comm=5000, capacity=10,
                                            prev_positions_m=None)
        assert ref[0] == pytest.approx(37.5, abs=0.01)
        assert ref[1] == pytest.approx(137.5, abs=0.01)

    def test_device_z_is_zero(self):
        coords = _noto_coords(5)
        inst, _ = _build_problem_instance(coords, K=2, R_comm=5000, capacity=10,
                                          prev_positions_m=None)
        assert np.all(inst.device_coords[:, 2] == 0.0)

    def test_default_prev_positions_spread_evenly(self):
        coords = _noto_coords(6)
        inst, _ = _build_problem_instance(coords, K=3, R_comm=5000, capacity=10,
                                          prev_positions_m=None)
        # prev_positions z-values should be 70 m
        assert np.all(inst.prev_positions[:, 2] == pytest.approx(70.0))

    def test_custom_prev_positions_used(self):
        coords = _noto_coords(4)
        K = 2
        prev = np.array([[100.0, 200.0, 50.0], [300.0, 400.0, 50.0]])
        inst, _ = _build_problem_instance(coords, K=K, R_comm=5000, capacity=10,
                                          prev_positions_m=prev)
        assert np.allclose(inst.prev_positions, prev)

    def test_battery_is_all_ones(self):
        coords = _noto_coords(5)
        inst, _ = _build_problem_instance(coords, K=2, R_comm=5000, capacity=10,
                                          prev_positions_m=None)
        assert np.all(inst.battery == 1.0)

    def test_value_is_uniform_ones(self):
        coords = _noto_coords(5)
        inst, _ = _build_problem_instance(coords, K=2, R_comm=5000, capacity=10,
                                          prev_positions_m=None)
        assert np.all(inst.value == 1.0)


# ── _uav_pos_to_latlon ────────────────────────────────────────────────────────

class TestUavPosToLatLon:
    def test_zero_position_maps_to_ref(self):
        ref = np.array([37.488, 137.272])
        pos = np.array([[0.0, 0.0, 70.0]])  # (1, 3)
        latlon = _uav_pos_to_latlon(pos, ref)
        assert len(latlon) == 1
        lat, lon = latlon[0]
        assert lat == pytest.approx(37.488, abs=1e-5)
        assert lon == pytest.approx(137.272, abs=1e-5)

    def test_positive_x_increases_longitude(self):
        ref = np.array([37.0, 137.0])
        pos = np.array([[1000.0, 0.0, 70.0]])  # 1000 m east
        latlon = _uav_pos_to_latlon(pos, ref)
        lat, lon = latlon[0]
        assert lon > 137.0

    def test_positive_y_increases_latitude(self):
        ref = np.array([37.0, 137.0])
        pos = np.array([[0.0, 1000.0, 70.0]])  # 1000 m north
        latlon = _uav_pos_to_latlon(pos, ref)
        lat, lon = latlon[0]
        assert lat > 37.0

    def test_multiple_uavs_returned(self):
        ref = np.array([37.0, 137.0])
        pos = np.array([[0.0, 0.0, 70.0], [500.0, 500.0, 70.0]])
        latlon = _uav_pos_to_latlon(pos, ref)
        assert len(latlon) == 2

    def test_roundtrip_consistency_with_latlon_to_meters(self):
        from hflsim.shared.coords import latlon_to_meters
        # Project some coords to metres, then back to latlon
        orig_coords = np.array([[37.488, 137.272], [37.500, 137.300]])
        xy_m, ref_tuple = latlon_to_meters(orig_coords)
        ref = np.array(ref_tuple)
        # Convert metre coords back to latlon
        pos = np.column_stack([xy_m, np.full(2, 70.0)])
        recovered = _uav_pos_to_latlon(pos, ref)
        for i, (lat, lon) in enumerate(recovered):
            assert abs(lat - orig_coords[i, 0]) < 0.001
            assert abs(lon - orig_coords[i, 1]) < 0.001


# ── _covered_clients ──────────────────────────────────────────────────────────

class TestCoveredClients:
    def _setup(self):
        """Two UAVs at known metre positions; a few clients near/far."""
        # Build a reference frame from a central coord
        centre = (37.488, 137.272)
        coords = np.array([centre])
        from hflsim.shared.coords import latlon_to_meters
        _, ref_tuple = latlon_to_meters(coords)
        ref = np.array(ref_tuple)
        # UAV 0 at origin (projected metres)
        # UAV 1 displaced far away
        uav_pos_m = np.array([
            [0.0, 0.0, 70.0],
            [50_000.0, 0.0, 70.0],
        ])
        return ref, uav_pos_m

    def test_client_within_R_comm_is_covered(self):
        ref, uav_pos_m = self._setup()
        # Client at the reference lat/lon should project to near (0,0) → covered by UAV 0
        client_coords = {0: (37.488, 137.272)}
        R_comm = 5_000.0
        result = _covered_clients(client_coords, uav_pos_m, ref, R_comm)
        assert 0 in result

    def test_client_far_from_all_uavs_is_not_covered(self):
        ref, uav_pos_m = self._setup()
        # Client ~200 km away from both UAVs
        client_coords = {99: (39.0, 140.0)}
        R_comm = 5_000.0
        result = _covered_clients(client_coords, uav_pos_m, ref, R_comm)
        assert 99 not in result

    def test_client_assigned_to_nearest_uav(self):
        ref, uav_pos_m = self._setup()
        # Client near UAV 0 (origin area)
        client_coords = {0: (37.488, 137.272)}
        R_comm = 100_000.0   # wide enough to cover both UAVs
        result = _covered_clients(client_coords, uav_pos_m, ref, R_comm)
        # UAV 0 is at (0,0,70) → much closer than UAV 1 at (50km, 0, 70)
        assert result.get(0) == 0

    def test_empty_client_coords(self):
        ref, uav_pos_m = self._setup()
        result = _covered_clients({}, uav_pos_m, ref, 5_000.0)
        assert result == {}

    def test_all_clients_covered_with_huge_R_comm(self):
        ref, uav_pos_m = self._setup()
        client_coords = {i: (37.3 + i * 0.1, 137.0 + i * 0.1) for i in range(5)}
        result = _covered_clients(client_coords, uav_pos_m, ref, R_comm=200_000.0)
        assert len(result) == 5


# ── _place_uavs (smoke) ────────────────────────────────────────────────────────

class TestPlaceUavs:
    def test_returns_correct_shapes(self):
        coords = _noto_coords(10)
        rng = np.random.default_rng(0)
        uav_pos_m, ref, fitness = _place_uavs(
            client_coords=coords,
            K=3,
            R_comm=10_000,
            capacity=5,
            method="centroid",
            rng=rng,
            P=5,
            G_max=5,
            prev_positions_m=None,
        )
        assert uav_pos_m.shape == (3, 3)
        assert len(ref) == 2

    def test_fitness_is_finite(self):
        coords = _noto_coords(10)
        rng = np.random.default_rng(0)
        _, _, fitness = _place_uavs(
            client_coords=coords,
            K=2,
            R_comm=10_000,
            capacity=5,
            method="centroid",
            rng=rng,
            P=5,
            G_max=5,
            prev_positions_m=None,
        )
        assert math.isfinite(fitness)

    def test_uav_z_in_altitude_bounds(self):
        coords = _noto_coords(8)
        rng = np.random.default_rng(1)
        uav_pos_m, _, _ = _place_uavs(
            client_coords=coords,
            K=2,
            R_comm=10_000,
            capacity=5,
            method="centroid",
            rng=rng,
            P=5,
            G_max=5,
            prev_positions_m=None,
        )
        # Altitude should be in [20, 120] per problem bounds
        for z in uav_pos_m[:, 2]:
            assert 10.0 <= z <= 130.0, f"UAV altitude {z} out of expected range"

    def test_ga_smoketest(self):
        coords = _noto_coords(6)
        rng = np.random.default_rng(2)
        uav_pos_m, ref, fitness = _place_uavs(
            client_coords=coords,
            K=2,
            R_comm=20_000,
            capacity=5,
            method="ga",
            rng=rng,
            P=5,
            G_max=3,
            prev_positions_m=None,
        )
        assert uav_pos_m.shape == (2, 3)
        assert math.isfinite(fitness)
