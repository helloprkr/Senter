#!/usr/bin/env python3
"""
State Persistence for Crash Recovery

Saves component state periodically and restores on restart.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger('senter.state')


class StateManager:
    """
    Manages state persistence for daemon components.

    Features:
    - Periodic checkpointing
    - Atomic file writes
    - State age warnings
    - Clean shutdown handling
    """

    def __init__(self, state_dir: str = None):
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            self.state_dir = Path(__file__).parent.parent / "data" / "state"

        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = 30  # seconds
        self.last_checkpoint = 0
        self.max_state_age = 3600  # 1 hour

    def save_component_state(self, component: str, state: dict) -> bool:
        """
        Save state for a specific component.

        Uses atomic write (write to temp, then rename) to prevent corruption.
        """
        component_file = self.state_dir / f"{component}.json"

        data = {
            "component": component,
            "state": state,
            "saved_at": time.time(),
            "version": 1
        }

        try:
            # Write atomically
            temp_file = component_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(data, indent=2, default=str))
            temp_file.rename(component_file)

            logger.debug(f"Saved state for {component}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state for {component}: {e}")
            return False

    def load_component_state(self, component: str) -> Optional[dict]:
        """
        Load state for a specific component.

        Returns None if no state exists or state is corrupted.
        """
        component_file = self.state_dir / f"{component}.json"

        if not component_file.exists():
            return None

        try:
            data = json.loads(component_file.read_text())
            age = time.time() - data.get("saved_at", 0)

            if age > self.max_state_age:
                logger.warning(
                    f"State for {component} is {age/3600:.1f} hours old - may be stale"
                )

            logger.info(f"Loaded state for {component} (age: {age:.0f}s)")
            return data.get("state", {})

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted state file for {component}: {e}")
            # Backup corrupted file
            backup = component_file.with_suffix('.corrupted')
            component_file.rename(backup)
            return None

        except Exception as e:
            logger.error(f"Failed to load state for {component}: {e}")
            return None

    def checkpoint(self, components: dict[str, Any]) -> int:
        """
        Checkpoint all component states.

        Returns number of components checkpointed.
        """
        now = time.time()

        if now - self.last_checkpoint < self.checkpoint_interval:
            return 0

        saved = 0
        for name, component in components.items():
            if hasattr(component, 'get_state'):
                try:
                    state = component.get_state()
                    if self.save_component_state(name, state):
                        saved += 1
                except Exception as e:
                    logger.error(f"Error getting state from {name}: {e}")

        self.last_checkpoint = now
        if saved > 0:
            logger.info(f"Checkpoint completed: {saved} components")

        return saved

    def restore_all(self, components: dict[str, Any]) -> int:
        """
        Restore state for all components.

        Returns number of components restored.
        """
        restored = 0

        for name, component in components.items():
            if hasattr(component, 'restore_state'):
                state = self.load_component_state(name)
                if state:
                    try:
                        component.restore_state(state)
                        restored += 1
                        logger.info(f"Restored state for {name}")
                    except Exception as e:
                        logger.error(f"Error restoring state for {name}: {e}")

        return restored

    def clear_state(self, component: str = None):
        """
        Clear state files.

        Args:
            component: Specific component to clear, or None for all
        """
        if component:
            state_file = self.state_dir / f"{component}.json"
            if state_file.exists():
                state_file.unlink()
                logger.info(f"Cleared state for {component}")
        else:
            for f in self.state_dir.glob("*.json"):
                f.unlink()
            logger.info("Cleared all state files")

    def get_state_info(self) -> dict:
        """Get information about stored state"""
        info = {"components": {}}

        for f in self.state_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                age = time.time() - data.get("saved_at", 0)
                info["components"][f.stem] = {
                    "age_seconds": age,
                    "saved_at": data.get("saved_at"),
                    "size_bytes": f.stat().st_size
                }
            except:
                info["components"][f.stem] = {"error": "corrupted"}

        return info

    def has_saved_state(self) -> bool:
        """Check if there is any saved state"""
        return any(self.state_dir.glob("*.json"))


# Mixin class for components to add state support
class StatefulComponent:
    """
    Mixin that adds state persistence to a component.

    Usage:
        class MyComponent(StatefulComponent):
            def get_state(self):
                return {"items": self.items}

            def restore_state(self, state):
                self.items = state.get("items", [])
    """

    def get_state(self) -> dict:
        """Override to return component state"""
        return {}

    def restore_state(self, state: dict):
        """Override to restore component state"""
        pass
