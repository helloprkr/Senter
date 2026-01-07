"""
Tests for Senter 3.0 Depth Improvement Features.

These tests cover the 8 critical gaps implemented via Ralph Wiggums.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# US-001: LLM-based context inference in ActivityMonitor
# ============================================================================

class TestActivityLLMContextInference:
    """Tests for LLM-based context inference in ActivityMonitor."""

    def test_llm_context_analysis_dataclass(self):
        """Test LLMContextAnalysis dataclass structure."""
        from intelligence.activity import LLMContextAnalysis

        analysis = LLMContextAnalysis(
            activity_type="coding",
            project_name="Senter",
            tasks=["debugging", "testing"],
            confidence=0.85,
            summary="User is coding in Python"
        )

        assert analysis.activity_type == "coding"
        assert analysis.project_name == "Senter"
        assert analysis.tasks == ["debugging", "testing"]
        assert analysis.confidence == 0.85
        assert "coding" in analysis.summary

    def test_llm_context_analysis_defaults(self):
        """Test LLMContextAnalysis default values."""
        from intelligence.activity import LLMContextAnalysis

        analysis = LLMContextAnalysis(activity_type="general")

        assert analysis.project_name is None
        assert analysis.tasks == []
        assert analysis.confidence == 0.5
        assert analysis.summary == ""

    @pytest.mark.asyncio
    async def test_infer_context_with_llm_fallback_no_model(self):
        """Test LLM inference falls back to heuristic when no model available."""
        from intelligence.activity import ActivityMonitor, LLMContextAnalysis

        monitor = ActivityMonitor(senter_engine=None)

        snapshot = {
            'app': 'VSCode',
            'window': 'main.py - Senter',
            'key_phrases': ['def test_function', 'import pytest']
        }

        result = await monitor.infer_context_with_llm(snapshot)

        assert isinstance(result, LLMContextAnalysis)
        assert result.activity_type == "coding"  # Heuristic should detect VSCode
        assert result.confidence == 0.3  # Low confidence for heuristic
        assert "Heuristic" in result.summary or "Fallback" in result.summary

    @pytest.mark.asyncio
    async def test_infer_context_with_llm_with_mock_model(self):
        """Test LLM inference with mocked model returns structured result."""
        from intelligence.activity import ActivityMonitor, LLMContextAnalysis

        # Create mock engine with mock model
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='{"activity_type": "coding", "project_name": "TestProject", "tasks": ["writing tests"], "confidence": 0.9, "summary": "User is writing unit tests"}')

        mock_engine = MagicMock()
        mock_engine.model = mock_model

        monitor = ActivityMonitor(senter_engine=mock_engine)

        snapshot = {
            'app': 'VSCode',
            'window': 'test_main.py - TestProject',
            'key_phrases': ['def test_function', 'assert result == expected']
        }

        result = await monitor.infer_context_with_llm(snapshot)

        assert isinstance(result, LLMContextAnalysis)
        assert result.activity_type == "coding"
        assert result.project_name == "TestProject"
        assert "writing tests" in result.tasks
        assert result.confidence == 0.9
        assert "unit tests" in result.summary

    @pytest.mark.asyncio
    async def test_infer_context_with_llm_handles_json_error(self):
        """Test LLM inference handles malformed JSON gracefully."""
        from intelligence.activity import ActivityMonitor, LLMContextAnalysis

        # Create mock engine with model that returns invalid JSON
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='Not valid JSON at all')

        mock_engine = MagicMock()
        mock_engine.model = mock_model

        monitor = ActivityMonitor(senter_engine=mock_engine)

        snapshot = {
            'app': 'Chrome',
            'window': 'Google Search',
            'key_phrases': ['search results']
        }

        result = await monitor.infer_context_with_llm(snapshot)

        # Should fall back to heuristic
        assert isinstance(result, LLMContextAnalysis)
        assert result.confidence == 0.3  # Fallback confidence
        assert "Fallback" in result.summary or "heuristic" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_infer_context_with_llm_handles_exception(self):
        """Test LLM inference handles model exceptions gracefully."""
        from intelligence.activity import ActivityMonitor, LLMContextAnalysis

        # Create mock engine with model that raises exception
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(side_effect=Exception("Model error"))

        mock_engine = MagicMock()
        mock_engine.model = mock_model

        monitor = ActivityMonitor(senter_engine=mock_engine)

        snapshot = {
            'app': 'Figma',
            'window': 'Design Project',
            'key_phrases': ['frame', 'component']
        }

        result = await monitor.infer_context_with_llm(snapshot)

        # Should fall back to heuristic
        assert isinstance(result, LLMContextAnalysis)
        assert result.activity_type == "design"  # Figma should be detected
        assert result.confidence == 0.3


# ============================================================================
# Context Inferencer (Heuristic) Tests
# ============================================================================

class TestContextInferencer:
    """Tests for heuristic context inference."""

    def test_infer_coding_context(self):
        """Test inferring coding context from VSCode."""
        from intelligence.activity import ContextInferencer

        inferencer = ContextInferencer()
        result = inferencer.infer_context({
            'app': 'VSCode',
            'window': 'main.py',
            'key_phrases': ['def function', 'import os']
        })

        assert result == 'coding'

    def test_infer_writing_context(self):
        """Test inferring writing context from Notion."""
        from intelligence.activity import ContextInferencer

        inferencer = ContextInferencer()
        result = inferencer.infer_context({
            'app': 'Notion',
            'window': 'Project Notes',
            'key_phrases': ['paragraph about', 'section heading']
        })

        assert result == 'writing'

    def test_infer_research_context(self):
        """Test inferring research context from browser with keywords."""
        from intelligence.activity import ContextInferencer

        inferencer = ContextInferencer()
        result = inferencer.infer_context({
            'app': 'Chrome',
            'window': 'arxiv.org - Paper Title',
            'key_phrases': ['abstract', 'research paper', 'study results']
        })

        assert result == 'research'

    def test_infer_general_context_unknown_app(self):
        """Test inferring general context for unknown apps."""
        from intelligence.activity import ContextInferencer

        inferencer = ContextInferencer()
        result = inferencer.infer_context({
            'app': 'UnknownApp',
            'window': 'Random Window',
            'key_phrases': []
        })

        assert result == 'general'


# ============================================================================
# ActivitySnapshot Tests
# ============================================================================

class TestActivitySnapshot:
    """Tests for ActivitySnapshot dataclass."""

    def test_activity_snapshot_with_llm_analysis(self):
        """Test ActivitySnapshot includes LLM analysis fields."""
        from intelligence.activity import ActivitySnapshot

        snapshot = ActivitySnapshot(
            timestamp=datetime.now(),
            active_app="VSCode",
            window_title="project/main.py",
            screen_text=["def main", "import sys"],
            inferred_context="coding",
            llm_analysis={"activity_type": "coding", "confidence": 0.9},
            detected_project="MyProject",
            detected_tasks=["debugging", "refactoring"]
        )

        assert snapshot.active_app == "VSCode"
        assert snapshot.inferred_context == "coding"
        assert snapshot.llm_analysis is not None
        assert snapshot.detected_project == "MyProject"
        assert len(snapshot.detected_tasks) == 2

    def test_activity_snapshot_defaults(self):
        """Test ActivitySnapshot default values."""
        from intelligence.activity import ActivitySnapshot

        snapshot = ActivitySnapshot(
            timestamp=datetime.now(),
            active_app="App",
            window_title="Window",
            screen_text=[],
            inferred_context="general"
        )

        assert snapshot.llm_analysis is None
        assert snapshot.detected_project is None
        assert snapshot.detected_tasks == []
