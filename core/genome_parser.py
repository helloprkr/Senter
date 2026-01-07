"""
Genome Parser - Loads and expands the genome.yaml configuration.

The genome is the DNA of Senter - all behavior emerges from it.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class GenomeParser:
    """Parses and manages the genome configuration."""

    def __init__(self, genome_path: Path):
        self.genome_path = Path(genome_path)
        self._genome: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load genome from YAML file."""
        if not self.genome_path.exists():
            raise FileNotFoundError(f"Genome not found: {self.genome_path}")

        with open(self.genome_path, "r") as f:
            content = f.read()

        # Expand environment variables ${VAR_NAME}
        content = self._expand_env_vars(content)

        self._genome = yaml.safe_load(content)

    def _expand_env_vars(self, content: str) -> str:
        """Expand ${VAR_NAME} patterns with environment variables."""
        pattern = r"\$\{([^}]+)\}"

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replacer, content)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value."""
        return self._genome.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation."""
        value = self._genome
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def version(self) -> str:
        """Get genome version."""
        return self._genome.get("version", "unknown")

    @property
    def name(self) -> str:
        """Get system name."""
        return self._genome.get("name", "Senter")

    @property
    def models(self) -> Dict[str, Any]:
        """Get model configurations."""
        return self._genome.get("models", {})

    @property
    def coupling(self) -> Dict[str, Any]:
        """Get coupling configuration."""
        return self._genome.get("coupling", {})

    @property
    def memory(self) -> Dict[str, Any]:
        """Get memory configuration."""
        return self._genome.get("memory", {})

    @property
    def evolution(self) -> Dict[str, Any]:
        """Get evolution configuration."""
        return self._genome.get("evolution", {})

    @property
    def knowledge(self) -> Dict[str, Any]:
        """Get knowledge configuration."""
        return self._genome.get("knowledge", {})

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get capabilities configuration."""
        return self._genome.get("capabilities", {})

    @property
    def interface(self) -> Dict[str, Any]:
        """Get interface configuration."""
        return self._genome.get("interface", {})

    def to_dict(self) -> Dict[str, Any]:
        """Return full genome as dictionary."""
        return self._genome.copy()

    def reload(self) -> None:
        """Reload genome from disk."""
        self._load()

    def save(self, path: Optional[Path] = None) -> None:
        """Save current genome to disk."""
        save_path = path or self.genome_path
        with open(save_path, "w") as f:
            yaml.dump(self._genome, f, default_flow_style=False, sort_keys=False)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update genome with new values (for mutations)."""
        self._deep_update(self._genome, updates)

    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


def load_genome(path: Path | str) -> GenomeParser:
    """Convenience function to load a genome."""
    return GenomeParser(Path(path))
