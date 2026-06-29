"""Rotary-wing UAV movement-energy model (reporting only).

Converts the abstract movement distance the optimizer minimizes into Joules and
a battery-fraction so the energy comparison is in physical units:

    E_move = P_fly * (d / v) + P_hover * t_serve

This is used only for *reporting* metrics, never inside the fitness, so the
optimizer's objective stays identical across methods. Constants are documented
defaults for a small rotary-wing UAV and are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyModel:
    """Configurable rotary-wing propulsion constants.

    Attributes
    ----------
    p_fly:
        Cruise (flight) power draw in watts.
    p_hover:
        Hover power draw in watts.
    cruise_speed:
        Horizontal cruise speed in m/s.
    t_serve:
        Service (hover) time per round in seconds.
    battery_capacity_j:
        Usable battery energy in Joules (for the battery-fraction conversion).
    """

    p_fly: float = 250.0
    p_hover: float = 200.0
    cruise_speed: float = 15.0
    t_serve: float = 60.0
    battery_capacity_j: float = 200_000.0

    def energy_joules(self, distance_m: float) -> float:
        """Energy (J) to fly ``distance_m`` then hover for ``t_serve``."""
        return self.p_fly * (distance_m / self.cruise_speed) + self.p_hover * self.t_serve

    def battery_fraction(self, distance_m: float) -> float:
        """Fraction of battery consumed for one reposition + service."""
        return self.energy_joules(distance_m) / self.battery_capacity_j


def movement_energy(
    distance_m: float, model: EnergyModel | None = None
) -> tuple[float, float]:
    """Return ``(joules, battery_fraction)`` for a reposition distance."""
    model = model or EnergyModel()
    return model.energy_joules(distance_m), model.battery_fraction(distance_m)
