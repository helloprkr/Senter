"""
Response Composer - Composes responses adapted to context.

Adapts to coupling mode, cognitive state, and conversation context.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.base import ModelInterface
    from coupling.protocols import CouplingMode
    from coupling.human_model import HumanCognitiveState
    from memory.living_memory import MemoryContext


@dataclass
class CompositionContext:
    """Context for response composition."""

    memory: Optional["MemoryContext"] = None
    knowledge: Optional[Dict[str, Any]] = None
    capabilities: List[str] = None
    joint_state: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, str]] = None  # Recent turns for context
    user_profile: List[Dict] = None  # Always-included user facts

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_profile is None:
            self.user_profile = []


class ResponseComposer:
    """
    Composes responses from intent and context.

    Adapts to:
    - Coupling mode (dialogue, teaching, directing, parallel)
    - Human cognitive state (frustration, time pressure, mode)
    - Available context (memory, knowledge)
    """

    def __init__(self, model: Optional["ModelInterface"] = None):
        self.model = model

    async def compose(
        self,
        intent: Dict[str, Any],
        context: CompositionContext,
        mode: Optional["CouplingMode"] = None,
        cognitive_state: Optional["HumanCognitiveState"] = None,
    ) -> str:
        """
        Compose a response given intent and context.

        Args:
            intent: Parsed user intent
            context: Memory and knowledge context
            mode: Current coupling mode
            cognitive_state: Inferred human cognitive state

        Returns:
            Composed response text
        """
        if not self.model:
            return self._compose_fallback(intent)

        # Build system prompt parts
        system_parts = [
            "You are Senter, a personal AI assistant that adapts to users.",
            "Be concise, helpful, and natural in conversation.",
        ]

        # Add mode-specific instructions
        if mode:
            mode_name = mode.name if hasattr(mode, "name") else str(mode)
            system_parts.append(f"Current interaction mode: {mode_name}")

            if mode_name == "TEACHING":
                system_parts.append("Explain your reasoning step by step.")
                system_parts.append("Check for understanding and provide examples.")
            elif mode_name == "DIRECTING":
                system_parts.append("Confirm understanding before acting.")
                system_parts.append("Be precise and report results clearly.")
            elif mode_name == "PARALLEL":
                system_parts.append("Acknowledge the task concisely.")
                system_parts.append("Propose sync points for checking in.")

        # Adapt to cognitive state
        if cognitive_state:
            if cognitive_state.frustration > 0.5:
                system_parts.append(
                    "The user seems frustrated. Be patient, empathetic, and solution-focused."
                )
            if cognitive_state.time_pressure in ("high", "urgent"):
                system_parts.append(
                    "The user is pressed for time. Be concise and get to the point."
                )
            if cognitive_state.energy_level < 0.4:
                system_parts.append(
                    "The user may be tired. Keep responses shorter and clearer."
                )

        # Add user profile (ALWAYS include - name, preferences, etc.)
        if context.user_profile:
            profile_info = self._format_user_profile(context.user_profile)
            if profile_info:
                system_parts.append(f"User profile: {profile_info}")

        # Add memory context
        if context.memory:
            memory_info = self._format_memory_context(context.memory)
            if memory_info:
                system_parts.append(f"Relevant memories: {memory_info}")

        # Add knowledge context
        if context.knowledge:
            knowledge_info = self._format_knowledge_context(context.knowledge)
            if knowledge_info:
                system_parts.append(f"Relevant knowledge: {knowledge_info}")

        # Build the prompt
        system_prompt = "\n".join(system_parts)
        user_input = intent.get("raw_input", intent.get("intent", ""))

        # Build conversation with history
        conversation_parts = []

        # Add conversation history (last N turns for context)
        if context.conversation_history:
            conversation_parts.append("Previous conversation:")
            for turn in context.conversation_history[-6:]:  # Last 6 turns (3 exchanges)
                role = "User" if turn.get("role") == "user" else "Senter"
                conversation_parts.append(f"{role}: {turn.get('content', '')}")
            conversation_parts.append("")  # Blank line

        # Add current user input
        conversation_parts.append(f"User: {user_input}")
        conversation_parts.append("")
        conversation_parts.append("Senter:")

        prompt = f"{system_prompt}\n\n" + "\n".join(conversation_parts)

        response = await self.model.generate(prompt)
        return response.strip()

    def _compose_fallback(self, intent: Dict[str, Any]) -> str:
        """Fallback response when no model is available."""
        user_input = intent.get("raw_input", intent.get("intent", ""))
        return f"I received your message: '{user_input[:50]}...'. Please configure a model to enable full responses."

    def _format_memory_context(self, memory: "MemoryContext") -> str:
        """Format memory context for the prompt."""
        parts = []

        if hasattr(memory, "semantic") and memory.semantic:
            facts = [str(s.get("content", s)) for s in memory.semantic[:3]]
            if facts:
                parts.append(f"Facts: {'; '.join(facts)}")

        if hasattr(memory, "episodic") and memory.episodic:
            recent = memory.episodic[:2]
            episodes = [f"'{e.input[:30]}...' -> '{e.response[:30]}...'" for e in recent]
            if episodes:
                parts.append(f"Recent: {'; '.join(episodes)}")

        if hasattr(memory, "affective") and memory.affective:
            if "avg_satisfaction" in memory.affective:
                parts.append(f"Satisfaction: {memory.affective['avg_satisfaction']:.1f}")

        return " | ".join(parts) if parts else ""

    def _format_knowledge_context(self, knowledge: Dict[str, Any]) -> str:
        """Format knowledge context for the prompt."""
        if not knowledge:
            return ""

        parts = []
        for key, value in list(knowledge.items())[:3]:
            if isinstance(value, str):
                parts.append(f"{key}: {value[:50]}")
            else:
                parts.append(f"{key}: {str(value)[:50]}")

        return " | ".join(parts) if parts else ""

    def _format_user_profile(self, profile: List[Dict]) -> str:
        """Format user profile facts for the prompt."""
        if not profile:
            return ""

        parts = []
        for fact in profile[:5]:  # Top 5 profile facts
            content = fact.get("content", "")
            domain = fact.get("domain", "")
            # Extract the key info
            if "name is" in content.lower():
                parts.append(f"Name: {content}")
            elif domain == "user_preference":
                parts.append(f"Preference: {content}")
            elif domain == "user_work":
                parts.append(f"Work: {content}")
            elif domain == "user_role":
                parts.append(f"Role: {content}")
            else:
                parts.append(content[:80])

        return " | ".join(parts) if parts else ""
