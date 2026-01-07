"""
Senter 3.0 Evolution Module

Enables the system to improve over time through:
- FitnessTracker: Measures how well the system is performing
- MutationEngine: Proposes configuration changes
- SelectionPressure: Determines which changes survive
"""

from .fitness import FitnessTracker, FitnessMetric
from .mutations import MutationEngine, Mutation
from .selection import SelectionPressure

__all__ = [
    "FitnessTracker",
    "FitnessMetric",
    "MutationEngine",
    "Mutation",
    "SelectionPressure",
]
