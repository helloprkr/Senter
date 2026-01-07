"""
Genome configuration validator.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import yaml


class ConfigValidator:
    """Validates genome.yaml configuration."""

    REQUIRED_SECTIONS = [
        'models',
        'memory',
        'coupling',
        'evolution'
    ]

    REQUIRED_MODEL_FIELDS = ['type']

    def __init__(self, genome_path: Path):
        self.genome_path = genome_path
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate configuration.

        Returns: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check file exists
        if not self.genome_path.exists():
            self.errors.append(f"Genome file not found: {self.genome_path}")
            return False, self.errors, self.warnings

        # Parse YAML
        try:
            with open(self.genome_path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML: {e}")
            return False, self.errors, self.warnings

        if config is None:
            self.errors.append("Empty configuration file")
            return False, self.errors, self.warnings

        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")

        # Validate models
        if 'models' in config:
            self._validate_models(config['models'])

        # Validate coupling
        if 'coupling' in config:
            self._validate_coupling(config['coupling'])

        # Validate evolution
        if 'evolution' in config:
            self._validate_evolution(config['evolution'])

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _validate_models(self, models: Dict):
        """Validate models configuration."""
        if 'primary' not in models:
            self.errors.append("Missing models.primary configuration")
            return

        primary = models['primary']
        if 'type' not in primary:
            self.errors.append("Missing models.primary.type")

        model_type = primary.get('type', '')
        if model_type == 'ollama':
            if 'model' not in primary:
                self.warnings.append("models.primary.model not set, will use default")
            if 'base_url' not in primary:
                self.warnings.append("models.primary.base_url not set, will use localhost:11434")
        elif model_type == 'gguf':
            if 'path' not in primary:
                self.errors.append("models.primary.path required for GGUF")
        elif model_type == 'openai':
            if 'model' not in primary:
                self.warnings.append("models.primary.model not set, will use gpt-4")

        # Check embeddings
        if 'embeddings' not in models:
            self.warnings.append("No embeddings model configured, semantic search will be limited")

    def _validate_coupling(self, coupling: Dict):
        """Validate coupling configuration."""
        if 'trust' in coupling:
            trust = coupling['trust']
            initial = trust.get('initial', 0.5)
            if not 0 <= initial <= 1:
                self.errors.append(f"coupling.trust.initial must be 0-1, got {initial}")

            trust_range = trust.get('range', [0, 1])
            if len(trust_range) != 2:
                self.errors.append("coupling.trust.range must have exactly 2 values")
            elif trust_range[0] >= trust_range[1]:
                self.errors.append("coupling.trust.range[0] must be less than range[1]")

    def _validate_evolution(self, evolution: Dict):
        """Validate evolution configuration."""
        if 'mutations' in evolution:
            mutations = evolution['mutations']
            rate = mutations.get('rate', 0.05)
            if not 0 <= rate <= 1:
                self.errors.append(f"evolution.mutations.rate must be 0-1, got {rate}")


def validate_config(genome_path: Path, quiet: bool = False) -> bool:
    """
    Validate and report config issues.

    Args:
        genome_path: Path to genome.yaml
        quiet: Suppress output

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator(genome_path)
    is_valid, errors, warnings = validator.validate()

    if not quiet:
        if warnings:
            for w in warnings:
                print(f"[WARN] {w}")

        if errors:
            for e in errors:
                print(f"[ERROR] {e}")
            return False

        print("[OK] Configuration valid")

    return is_valid
