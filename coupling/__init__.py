"""
Senter 3.0 Coupling Module

The key innovation: Bidirectional cognitive coupling between human and AI.

Components:
- JointState: Shared cognitive space
- HumanModel: AI's model of the human
- CouplingProtocols: Different interaction modes
- TrustTracker: Trust relationship tracking
"""

from .joint_state import JointState, Goal
from .human_model import HumanModel, HumanCognitiveState, HumanProfile
from .protocols import CouplingFacilitator, CouplingMode, Protocol
from .trust import TrustTracker

__all__ = [
    "JointState",
    "Goal",
    "HumanModel",
    "HumanCognitiveState",
    "HumanProfile",
    "CouplingFacilitator",
    "CouplingMode",
    "Protocol",
    "TrustTracker",
]
