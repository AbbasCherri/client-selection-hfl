"""Shared problem definition: instance, value, assignment, fitness, energy."""

from .instance import ProblemInstance, generate_instance
from .value import compute_value, beta_schedule
from .assignment import greedy_assignment, AssignmentResult
from .fitness import Fitness, fitness_components
from .energy import movement_energy

__all__ = [
    "ProblemInstance",
    "generate_instance",
    "compute_value",
    "beta_schedule",
    "greedy_assignment",
    "AssignmentResult",
    "Fitness",
    "fitness_components",
    "movement_energy",
]
