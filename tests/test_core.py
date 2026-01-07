"""
Tests for core module.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from core.genome_parser import GenomeParser, load_genome
from core.intent import IntentParser, Intent
from core.composer import ResponseComposer, CompositionContext


class TestGenomeParser:
    """Tests for GenomeParser."""

    def test_load_genome(self, tmp_path):
        """Test loading a genome file."""
        genome_content = {
            "version": "3.0",
            "name": "TestSenter",
            "models": {
                "primary": {"type": "openai", "model": "gpt-4o-mini"},
            },
        }

        genome_file = tmp_path / "genome.yaml"
        with open(genome_file, "w") as f:
            yaml.dump(genome_content, f)

        parser = load_genome(genome_file)

        assert parser.version == "3.0"
        assert parser.name == "TestSenter"
        assert parser.models["primary"]["type"] == "openai"

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        """Test environment variable expansion."""
        monkeypatch.setenv("TEST_MODEL_PATH", "/path/to/model")

        genome_content = """
version: "3.0"
models:
  primary:
    path: "${TEST_MODEL_PATH}"
"""
        genome_file = tmp_path / "genome.yaml"
        genome_file.write_text(genome_content)

        parser = load_genome(genome_file)
        assert parser.models["primary"]["path"] == "/path/to/model"

    def test_nested_get(self, tmp_path):
        """Test nested configuration access."""
        genome_content = {
            "coupling": {
                "trust": {"initial": 0.5, "range": [0, 1]},
            },
        }

        genome_file = tmp_path / "genome.yaml"
        with open(genome_file, "w") as f:
            yaml.dump(genome_content, f)

        parser = load_genome(genome_file)
        assert parser.get_nested("coupling", "trust", "initial") == 0.5


class TestIntentParser:
    """Tests for IntentParser."""

    def test_heuristic_parsing(self):
        """Test heuristic intent parsing without model."""
        parser = IntentParser(model=None)

        # Test synchronous fallback
        intent = parser._parse_heuristic("Help me debug this error!", None)

        assert intent.tone in ("frustrated", "neutral")
        assert "web_search" not in intent.capabilities_needed

    def test_capability_detection(self):
        """Test capability detection from triggers."""
        parser = IntentParser(model=None)

        # Web search triggers
        intent = parser._parse_heuristic("What's the latest news today?", None)
        assert "web_search" in intent.capabilities_needed

        # Remember triggers
        intent = parser._parse_heuristic("Remember that my birthday is March 5", None)
        assert "remember" in intent.capabilities_needed

        # Recall triggers
        intent = parser._parse_heuristic("Do you remember my birthday?", None)
        assert "recall" in intent.capabilities_needed

    def test_urgency_detection(self):
        """Test urgency level detection."""
        parser = IntentParser(model=None)

        # High urgency
        intent = parser._parse_heuristic("I need this done ASAP!", None)
        assert intent.urgency == "high"

        # Low urgency
        intent = parser._parse_heuristic("No rush, but can you help whenever?", None)
        assert intent.urgency == "low"


class TestResponseComposer:
    """Tests for ResponseComposer."""

    def test_fallback_response(self):
        """Test fallback when no model available."""
        composer = ResponseComposer(model=None)

        intent = {"raw_input": "Hello, how are you?"}
        context = CompositionContext()

        # Sync call to fallback
        response = composer._compose_fallback(intent)

        assert "Hello" in response
        assert "configure a model" in response

    def test_context_formatting(self):
        """Test context formatting for prompts."""
        composer = ResponseComposer(model=None)

        # Test knowledge formatting
        knowledge = {"topic": "Python programming", "level": "advanced"}
        formatted = composer._format_knowledge_context(knowledge)

        assert "topic" in formatted
        assert "Python" in formatted
