"""Tests for haversine and latlon_to_meters (hflsim/shared/coords.py)."""

import math

import numpy as np
import pytest

from hflsim.shared.coords import haversine, latlon_to_meters


# ── haversine ─────────────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine((37.488, 137.272), (37.488, 137.272)) == pytest.approx(0.0)

    def test_symmetric(self):
        a = (37.0, 137.0)
        b = (38.0, 138.0)
        assert haversine(a, b) == pytest.approx(haversine(b, a))

    def test_equator_1_degree_longitude(self):
        # At the equator, 1° longitude ≈ 2π * 6,371,000 / 360 ≈ 111,194.9 m
        d = haversine((0.0, 0.0), (0.0, 1.0))
        assert 111_000 < d < 111_500

    def test_1_degree_latitude(self):
        # 1° latitude is approximately 111,195 m anywhere (not affected by cos)
        d = haversine((37.0, 137.0), (38.0, 137.0))
        assert 110_500 < d < 111_700

    def test_positive_distance(self):
        assert haversine((37.0, 137.0), (38.5, 138.5)) > 0.0

    def test_known_distance_noto_region(self):
        # Two points about 55 km apart in the Noto Peninsula area
        a = (37.0, 136.9)
        b = (37.5, 137.3)
        d = haversine(a, b)
        assert 50_000 < d < 80_000, f"Expected ~55-65 km, got {d:.0f} m"

    def test_antipodal_points(self):
        d = haversine((0.0, 0.0), (0.0, 180.0))
        # Half circumference ≈ 20,015,087 m
        assert 19_900_000 < d < 20_100_000


# ── latlon_to_meters ──────────────────────────────────────────────────────────

class TestLatLonToMeters:
    def test_ref_point_maps_to_origin(self):
        coords = np.array([[37.488, 137.272]])
        xy, ref = latlon_to_meters(coords)
        assert xy[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert xy[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert ref == (pytest.approx(37.488), pytest.approx(137.272))

    def test_output_shape(self):
        coords = np.array([[37.0, 137.0], [38.0, 138.0], [37.5, 137.5]])
        xy, ref = latlon_to_meters(coords)
        assert xy.shape == (3, 2)

    def test_custom_ref(self):
        coords = np.array([[37.0, 137.0]])
        ref_in = (37.0, 137.0)
        xy, ref_out = latlon_to_meters(coords, ref=ref_in)
        assert xy[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert xy[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert ref_out == ref_in

    def test_y_scale_approx_111km_per_degree(self):
        coords = np.array([[37.0, 137.0], [38.0, 137.0]])
        xy, _ = latlon_to_meters(coords)
        dy = abs(xy[1, 1] - xy[0, 1])
        assert 110_000 < dy < 112_000

    def test_x_scale_reduced_by_cos_at_latitude(self):
        # At latitude 37°, 1° longitude < 1° latitude
        coords = np.array([[37.0, 137.0], [37.0, 138.0]])
        xy, _ = latlon_to_meters(coords)
        dx = abs(xy[1, 0] - xy[0, 0])
        # cos(37°) ≈ 0.7986 → ~88,800 m
        assert 85_000 < dx < 92_000

    def test_symmetric_about_mean(self):
        # Symmetric pairs should produce symmetric x/y values
        coords = np.array([[37.0, 137.0], [38.0, 137.0]])  # symmetric around 37.5
        xy, _ = latlon_to_meters(coords)
        assert xy[0, 1] == pytest.approx(-xy[1, 1], abs=1e-6)

    def test_consistent_with_haversine(self):
        # Project two nearby points and compare Euclidean vs Haversine distance
        a = (37.488, 137.272)
        b = (37.510, 137.290)
        coords = np.array([a, b])
        xy, _ = latlon_to_meters(coords)
        eucl = float(np.sqrt(np.sum((xy[0] - xy[1])**2)))
        hav = haversine(a, b)
        # Within 0.1% for short distances
        assert abs(eucl - hav) / hav < 0.001

    def test_returns_ref_as_mean_of_input(self):
        coords = np.array([[36.0, 136.0], [38.0, 138.0]])
        _, ref = latlon_to_meters(coords)
        assert ref[0] == pytest.approx(37.0)
        assert ref[1] == pytest.approx(137.0)
