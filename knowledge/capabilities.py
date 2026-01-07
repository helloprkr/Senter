"""
Capability Registry - Tracks available capabilities.

Manages what the system can do and how to invoke capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Capability:
    """A registered capability."""

    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    handler: Optional[Callable] = None
    always_available: bool = False
    enabled: bool = True
    usage_count: int = 0
    success_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "always_available": self.always_available,
            "enabled": self.enabled,
            "usage_count": self.usage_count,
            "success_rate": (
                self.success_count / self.usage_count if self.usage_count > 0 else 0
            ),
        }


class CapabilityRegistry:
    """
    Registry of available capabilities.

    Tracks:
    - Built-in capabilities (respond, remember, recall)
    - Tool-based capabilities (web_search, file_ops)
    - Discovered capabilities (auto-discovered from tools/)
    """

    def __init__(self, config: Dict = None):
        config = config or {}
        self._capabilities: Dict[str, Capability] = {}

        # Load built-in capabilities
        builtin = config.get("builtin", [])
        for cap_config in builtin:
            self.register(
                name=cap_config.get("name", "unknown"),
                description=cap_config.get("description", ""),
                triggers=cap_config.get("triggers", []),
                always_available=cap_config.get("always_available", False),
            )

        # Discovery settings
        self.discovery_enabled = config.get("discovery", {}).get("enabled", False)
        self.discovery_sources = config.get("discovery", {}).get("sources", [])

    def register(
        self,
        name: str,
        description: str,
        triggers: List[str] = None,
        handler: Callable = None,
        always_available: bool = False,
    ) -> Capability:
        """
        Register a capability.

        Args:
            name: Capability name
            description: What it does
            triggers: Keywords that trigger this capability
            handler: Function to execute the capability
            always_available: Whether it's always active

        Returns:
            Registered capability
        """
        capability = Capability(
            name=name,
            description=description,
            triggers=triggers or [],
            handler=handler,
            always_available=always_available,
        )

        self._capabilities[name] = capability
        return capability

    def get(self, name: str) -> Optional[Capability]:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def get_available(self, intent: Dict[str, Any] = None) -> List[str]:
        """
        Get available capabilities for an intent.

        Args:
            intent: Parsed user intent

        Returns:
            List of available capability names
        """
        available = []

        for cap in self._capabilities.values():
            if not cap.enabled:
                continue

            if cap.always_available:
                available.append(cap.name)
                continue

            # Check triggers against intent
            if intent:
                input_text = intent.get("raw_input", "").lower()
                if any(trigger in input_text for trigger in cap.triggers):
                    available.append(cap.name)

        return available

    def get_available_names(self) -> List[str]:
        """Get names of all available capabilities."""
        return [cap.name for cap in self._capabilities.values() if cap.enabled]

    def get_by_trigger(self, text: str) -> List[Capability]:
        """Get capabilities that match text triggers."""
        text_lower = text.lower()
        matching = []

        for cap in self._capabilities.values():
            if not cap.enabled:
                continue

            if any(trigger in text_lower for trigger in cap.triggers):
                matching.append(cap)

        return matching

    async def execute(
        self,
        name: str,
        context: Dict[str, Any] = None,
    ) -> Optional[Any]:
        """
        Execute a capability.

        Args:
            name: Capability name
            context: Execution context

        Returns:
            Capability result or None
        """
        cap = self._capabilities.get(name)
        if not cap or not cap.handler or not cap.enabled:
            return None

        cap.usage_count += 1

        try:
            result = await cap.handler(context or {})
            cap.success_count += 1
            return result
        except Exception:
            return None

    def set_handler(self, name: str, handler: Callable) -> bool:
        """Set handler for a capability."""
        cap = self._capabilities.get(name)
        if cap:
            cap.handler = handler
            return True
        return False

    def enable(self, name: str) -> bool:
        """Enable a capability."""
        cap = self._capabilities.get(name)
        if cap:
            cap.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a capability."""
        cap = self._capabilities.get(name)
        if cap:
            cap.enabled = False
            return True
        return False

    def discover(self) -> List[str]:
        """
        Discover capabilities from configured sources.

        Returns:
            List of newly discovered capability names
        """
        if not self.discovery_enabled:
            return []

        discovered = []

        for source in self.discovery_sources:
            path = source.get("path", "")
            pattern = source.get("pattern", "*.py")

            # In a real implementation, this would scan the filesystem
            # and introspect Python files for capability definitions

        return discovered

    def get_stats(self) -> Dict[str, Any]:
        """Get capability statistics."""
        enabled = sum(1 for c in self._capabilities.values() if c.enabled)
        total_usage = sum(c.usage_count for c in self._capabilities.values())
        total_success = sum(c.success_count for c in self._capabilities.values())

        return {
            "total": len(self._capabilities),
            "enabled": enabled,
            "total_usage": total_usage,
            "overall_success_rate": (
                total_success / total_usage if total_usage > 0 else 0
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "capabilities": {
                name: cap.to_dict() for name, cap in self._capabilities.items()
            },
            "stats": self.get_stats(),
        }
