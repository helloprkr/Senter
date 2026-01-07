"""
Coupling Protocols - Different modes of human-AI interaction.

The key insight: Different tasks need different coupling patterns.
A debugging session is different from learning, which is different
from creative exploration.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .joint_state import JointState


class CouplingMode(Enum):
    """The different modes of human-AI coupling."""

    DIALOGUE = auto()  # Turn-taking conversation
    PARALLEL = auto()  # Both working, sync periodically
    TEACHING = auto()  # AI explains, human learns
    DIRECTING = auto()  # Human guides, AI executes


@dataclass
class Protocol:
    """A coupling protocol definition."""

    name: str
    mode: CouplingMode
    triggers: List[str]
    behaviors: List[str]
    description: str = ""


class CouplingFacilitator:
    """
    Manages coupling between human and AI.

    Selects appropriate protocol based on context,
    applies protocol-specific modifications to responses.
    """

    DEFAULT_PROTOCOLS = [
        Protocol(
            name="dialogue",
            mode=CouplingMode.DIALOGUE,
            triggers=["default", "chat", "talk", "discuss"],
            behaviors=["respond", "clarify", "suggest"],
            description="Turn-taking conversation",
        ),
        Protocol(
            name="parallel",
            mode=CouplingMode.PARALLEL,
            triggers=["research", "deep work", "while you", "in the meantime"],
            behaviors=["acknowledge", "work_background", "sync"],
            description="Both working, periodic sync",
        ),
        Protocol(
            name="teaching",
            mode=CouplingMode.TEACHING,
            triggers=["explain", "teach", "how does", "why", "learn"],
            behaviors=["explain_reasoning", "check_understanding", "examples"],
            description="AI explains, human learns",
        ),
        Protocol(
            name="directing",
            mode=CouplingMode.DIRECTING,
            triggers=["do", "run", "execute", "implement", "create"],
            behaviors=["confirm", "execute", "report"],
            description="Human guides, AI executes",
        ),
    ]

    def __init__(self, protocol_configs: List[Dict] = None):
        self.protocols = self.DEFAULT_PROTOCOLS.copy()
        self.current_mode = CouplingMode.DIALOGUE
        self.current_protocol = self.protocols[0]

        # Parse config if provided
        if protocol_configs:
            self._parse_configs(protocol_configs)

    def _parse_configs(self, configs: List[Dict]) -> None:
        """Parse protocol configurations from genome."""
        for config in configs:
            name = config.get("name", "dialogue")
            mode_map = {
                "dialogue": CouplingMode.DIALOGUE,
                "parallel": CouplingMode.PARALLEL,
                "teaching": CouplingMode.TEACHING,
                "directing": CouplingMode.DIRECTING,
            }
            mode = mode_map.get(name.lower(), CouplingMode.DIALOGUE)

            protocol = Protocol(
                name=name,
                mode=mode,
                triggers=config.get("triggers", []),
                behaviors=config.get("behavior", []),
                description=config.get("description", ""),
            )

            # Replace or add
            existing = [i for i, p in enumerate(self.protocols) if p.name == name]
            if existing:
                self.protocols[existing[0]] = protocol
            else:
                self.protocols.append(protocol)

    def select_mode(
        self,
        input_text: str,
        joint_state: Optional["JointState"] = None,
    ) -> CouplingMode:
        """Select appropriate coupling mode based on input and state."""
        input_lower = input_text.lower()

        # Check each protocol's triggers
        for protocol in self.protocols:
            for trigger in protocol.triggers:
                if trigger in input_lower:
                    self.current_mode = protocol.mode
                    self.current_protocol = protocol
                    return protocol.mode

        # Consider joint state context
        if joint_state:
            # If there are active goals, prefer directing mode
            active_goals = joint_state.get_active_goals()
            if active_goals and len(input_text) < 50:
                self.current_mode = CouplingMode.DIRECTING
                self.current_protocol = self._get_protocol_by_mode(CouplingMode.DIRECTING)
                return CouplingMode.DIRECTING

        # Default to dialogue
        self.current_mode = CouplingMode.DIALOGUE
        self.current_protocol = self.protocols[0]
        return CouplingMode.DIALOGUE

    def _get_protocol_by_mode(self, mode: CouplingMode) -> Protocol:
        """Get protocol by mode."""
        for protocol in self.protocols:
            if protocol.mode == mode:
                return protocol
        return self.protocols[0]

    def apply_protocol(
        self,
        response: str,
        mode: CouplingMode,
        joint_state: Optional["JointState"] = None,
    ) -> str:
        """Apply protocol-specific modifications to response."""
        if mode == CouplingMode.TEACHING:
            response = self._apply_teaching_protocol(response)
        elif mode == CouplingMode.PARALLEL:
            response = self._apply_parallel_protocol(response)
        elif mode == CouplingMode.DIRECTING:
            response = self._apply_directing_protocol(response)

        return response

    def _apply_teaching_protocol(self, response: str) -> str:
        """Apply teaching protocol modifications."""
        # Add reasoning explanation if not present
        reasoning_words = ["because", "reason", "since", "therefore"]
        has_reasoning = any(word in response.lower() for word in reasoning_words)

        if not has_reasoning and len(response) < 300:
            response = response + "\n\nWould you like me to explain my reasoning?"

        return response

    def _apply_parallel_protocol(self, response: str) -> str:
        """Apply parallel work protocol modifications."""
        sync_phrase = "\n\n[I'll continue working on this. Let me know when you want to sync up.]"
        if sync_phrase.strip() not in response:
            response = response + sync_phrase

        return response

    def _apply_directing_protocol(self, response: str) -> str:
        """Apply directing protocol modifications."""
        confirmation_starters = ["I'll", "I will", "Understood", "Got it", "On it"]
        starts_with_confirmation = any(
            response.startswith(starter) for starter in confirmation_starters
        )

        if not starts_with_confirmation:
            response = f"Understood. {response}"

        return response

    def get_mode_name(self) -> str:
        """Get current mode name."""
        return self.current_mode.name

    def get_protocol_behaviors(self) -> List[str]:
        """Get behaviors for current protocol."""
        return self.current_protocol.behaviors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_mode": self.current_mode.name,
            "current_protocol": self.current_protocol.name,
            "behaviors": self.current_protocol.behaviors,
            "available_protocols": [p.name for p in self.protocols],
        }
