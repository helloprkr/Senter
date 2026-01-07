"""
Context Engine - Manages current operational context.

Tracks conversation state, active focus, and operational context.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # user | assistant
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextEngine:
    """
    Manages operational context.

    Tracks:
    - Current conversation history
    - Active focus/topic
    - Session metadata
    - Relevant context for the LLM
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history

        # Conversation history
        self._history: List[ConversationTurn] = []

        # Current focus
        self.focus: Optional[str] = None
        self.focus_started: Optional[datetime] = None

        # Session metadata
        self.session_started = datetime.now()
        self.turn_count = 0

        # Context variables
        self._variables: Dict[str, Any] = {}

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None,
    ) -> ConversationTurn:
        """
        Add a conversation turn.

        Args:
            role: "user" or "assistant"
            content: Turn content
            metadata: Additional metadata

        Returns:
            Created turn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        self._history.append(turn)
        self.turn_count += 1

        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        return turn

    def get_history(self, limit: int = None) -> List[ConversationTurn]:
        """Get conversation history."""
        if limit:
            return self._history[-limit:]
        return self._history.copy()

    def get_history_for_llm(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get history formatted for LLM context.

        Returns:
            List of {role, content} dicts
        """
        history = self.get_history(limit)
        return [{"role": t.role, "content": t.content} for t in history]

    def set_focus(self, focus: str) -> None:
        """Set current focus/topic."""
        self.focus = focus
        self.focus_started = datetime.now()

    def clear_focus(self) -> None:
        """Clear current focus."""
        self.focus = None
        self.focus_started = None

    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self._variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self._variables.get(key, default)

    def clear_variable(self, key: str) -> None:
        """Clear a context variable."""
        if key in self._variables:
            del self._variables[key]

    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        focus_duration = None
        if self.focus_started:
            focus_duration = (datetime.now() - self.focus_started).total_seconds()

        return {
            "focus": self.focus,
            "focus_duration_seconds": focus_duration,
            "turn_count": self.turn_count,
            "history_size": len(self._history),
            "session_duration_seconds": (
                datetime.now() - self.session_started
            ).total_seconds(),
            "variables": list(self._variables.keys()),
        }

    def build_context_prompt(self, max_tokens: int = 2000) -> str:
        """
        Build a context prompt for the LLM.

        Args:
            max_tokens: Maximum context size (approximate)

        Returns:
            Context prompt string
        """
        parts = []

        # Add focus if set
        if self.focus:
            parts.append(f"Current focus: {self.focus}")

        # Add relevant variables
        if self._variables:
            var_parts = []
            for key, value in list(self._variables.items())[:5]:
                if isinstance(value, str) and len(value) < 100:
                    var_parts.append(f"  {key}: {value}")
            if var_parts:
                parts.append("Context variables:\n" + "\n".join(var_parts))

        # Add recent history
        history = self.get_history_for_llm(limit=5)
        if history:
            history_parts = []
            for turn in history:
                role = "User" if turn["role"] == "user" else "Assistant"
                content = turn["content"][:200]  # Truncate long turns
                history_parts.append(f"{role}: {content}")
            parts.append("Recent conversation:\n" + "\n".join(history_parts))

        return "\n\n".join(parts)

    def reset(self) -> None:
        """Reset context state."""
        self._history.clear()
        self.focus = None
        self.focus_started = None
        self.turn_count = 0
        self.session_started = datetime.now()
        self._variables.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.get_summary(),
            "history": [
                {"role": t.role, "content": t.content, "timestamp": t.timestamp.isoformat()}
                for t in self._history[-5:]
            ],
            "variables": self._variables.copy(),
        }
