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


# ============================================================================
# US-002: Project Detection in ActivityMonitor
# ============================================================================

class TestProjectDetection:
    """Tests for project detection in ActivityMonitor."""

    def test_project_detector_vscode_pattern(self):
        """Test detecting project from VSCode window title."""
        from intelligence.activity import ProjectDetector

        detector = ProjectDetector()

        # VSCode pattern: "file.py - ProjectName - Visual Studio Code"
        result = detector.detect_project("main.py - MyProject - Visual Studio Code")
        assert result == "MyProject"

    def test_project_detector_pycharm_pattern(self):
        """Test detecting project from PyCharm window title."""
        from intelligence.activity import ProjectDetector

        detector = ProjectDetector()

        # PyCharm pattern: "file.py – ProjectName"
        result = detector.detect_project("test.py – SenterProject")
        assert result == "SenterProject"

    def test_project_detector_path_pattern(self):
        """Test detecting project from file path in title."""
        from intelligence.activity import ProjectDetector

        detector = ProjectDetector()

        # Path pattern: "/path/to/ProjectName/src/file.py"
        result = detector.detect_project("/Users/dev/MyApp/src/main.py")
        assert result == "MyApp"

    def test_project_detector_terminal_pattern(self):
        """Test detecting project from terminal path."""
        from intelligence.activity import ProjectDetector

        detector = ProjectDetector()

        # Terminal pattern: "~/Projects/ProjectName"
        result = detector.detect_project("~/Projects/CoolProject")
        assert result == "CoolProject"

    def test_project_detector_ignores_common_words(self):
        """Test that common words are ignored as project names."""
        from intelligence.activity import ProjectDetector

        detector = ProjectDetector()

        # Should not detect 'Untitled' or 'home'
        result = detector.detect_project("Untitled - Visual Studio Code")
        assert result is None

    def test_project_detector_history(self):
        """Test project detection history tracking."""
        from intelligence.activity import ProjectDetector

        detector = ProjectDetector()

        # Detect same project multiple times
        detector.detect_project("main.py - TestProject - Visual Studio Code")
        detector.detect_project("test.py - TestProject - Visual Studio Code")
        detector.detect_project("app.py - TestProject - Visual Studio Code")

        history = detector.get_most_common_project()
        assert len(history) > 0
        assert history[0][0] == "TestProject"
        assert history[0][1] == 3

    def test_project_detector_no_match(self):
        """Test when no project can be detected."""
        from intelligence.activity import ProjectDetector

        detector = ProjectDetector()

        result = detector.detect_project("Google Chrome")
        assert result is None

    def test_activity_monitor_has_project_detector(self):
        """Test ActivityMonitor initializes with ProjectDetector."""
        from intelligence.activity import ActivityMonitor, ProjectDetector

        monitor = ActivityMonitor()

        assert hasattr(monitor, 'project_detector')
        assert isinstance(monitor.project_detector, ProjectDetector)
        assert hasattr(monitor, 'detected_projects')

    def test_activity_monitor_get_current_project(self):
        """Test getting current project from ActivityMonitor."""
        from intelligence.activity import ActivityMonitor, ActivitySnapshot

        monitor = ActivityMonitor()

        # Add snapshot with project
        snapshot = ActivitySnapshot(
            timestamp=datetime.now(),
            active_app="VSCode",
            window_title="main.py - TestProject - Visual Studio Code",
            screen_text=[],
            inferred_context="coding",
            detected_project="TestProject"
        )
        monitor.history.append(snapshot)

        assert monitor.get_current_project() == "TestProject"

    def test_activity_monitor_get_project_history(self):
        """Test getting project history from ActivityMonitor."""
        from intelligence.activity import ActivityMonitor

        monitor = ActivityMonitor()

        # Manually add project detections
        monitor.detected_projects["ProjectA"] = 5
        monitor.detected_projects["ProjectB"] = 3

        history = monitor.get_project_history()
        assert history["ProjectA"] == 5
        assert history["ProjectB"] == 3

    def test_activity_monitor_get_snapshots_for_project(self):
        """Test getting snapshots for a specific project."""
        from intelligence.activity import ActivityMonitor, ActivitySnapshot

        monitor = ActivityMonitor()

        # Add snapshots for different projects
        for i in range(5):
            monitor.history.append(ActivitySnapshot(
                timestamp=datetime.now(),
                active_app="VSCode",
                window_title=f"file{i}.py",
                screen_text=[],
                inferred_context="coding",
                detected_project="ProjectA" if i % 2 == 0 else "ProjectB"
            ))

        projectA_snaps = monitor.get_snapshots_for_project("ProjectA")
        assert len(projectA_snaps) == 3  # Indexes 0, 2, 4

    def test_activity_summary_includes_projects(self):
        """Test that activity summary includes project information."""
        from intelligence.activity import ActivityMonitor, ActivitySnapshot

        monitor = ActivityMonitor()

        # Add snapshots with projects
        monitor.history.append(ActivitySnapshot(
            timestamp=datetime.now(),
            active_app="VSCode",
            window_title="main.py",
            screen_text=[],
            inferred_context="coding",
            detected_project="MyProject"
        ))

        summary = monitor.get_activity_summary()

        assert 'top_projects' in summary
        assert 'current_project' in summary
        assert summary['current_project'] == "MyProject"
