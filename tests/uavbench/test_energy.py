"""EnergyModel: exact formulas against hand-computed values."""

import pytest

from uavbench.problem.energy import EnergyModel


def test_zero_distance_is_pure_hover():
    m = EnergyModel()
    assert m.energy_joules(0.0) == m.p_hover * m.t_serve


def test_energy_linear_in_distance():
    m = EnergyModel()
    e0, e1, e2 = m.energy_joules(0.0), m.energy_joules(150.0), m.energy_joules(300.0)
    assert e2 - e1 == pytest.approx(e1 - e0)
    # Slope is P_fly / cruise_speed joules per metre.
    assert e1 - e0 == pytest.approx(m.p_fly * 150.0 / m.cruise_speed)


def test_hand_computed_default_values():
    # Defaults: p_fly=250 W, p_hover=200 W, cruise=15 m/s, t_serve=60 s.
    m = EnergyModel()
    assert m.energy_joules(1500.0) == pytest.approx(250.0 * 100.0 + 200.0 * 60.0)


def test_battery_fraction_is_energy_over_capacity():
    m = EnergyModel()
    d = 4321.0
    assert m.battery_fraction(d) == pytest.approx(m.energy_joules(d) / m.battery_capacity_j)
