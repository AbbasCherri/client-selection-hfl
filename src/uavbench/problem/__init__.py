"""Shared problem definition: instance, value, assignment, fitness, energy."""

from .assignment import AssignmentResult, greedy_assignment
from .energy import movement_energy
from .fitness import Fitness, fitness_components
from .instance import ProblemInstance, generate_instance
from .value import beta_schedule, compute_value

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
