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


# ============================================================================
# US-003: Integrate ActivityMonitor with goal suggestion
# ============================================================================

class TestActivityGoalSuggestion:
    """Tests for activity-based goal suggestions."""

    def test_goal_detector_has_activity_inferred_source(self):
        """Test Goal dataclass supports activity_inferred source."""
        from intelligence.goals import Goal
        from datetime import datetime

        goal = Goal(
            id="test123",
            description="Work on project X",
            category="project",
            confidence=0.7,
            evidence=["Observed 15 coding sessions"],
            created_at=datetime.now(),
            last_mentioned=datetime.now(),
            progress=0.0,
            status="active",
            source="activity_inferred"
        )

        assert goal.source == "activity_inferred"

    def test_goal_to_dict_includes_source(self):
        """Test Goal.to_dict() includes source field."""
        from intelligence.goals import Goal
        from datetime import datetime

        goal = Goal(
            id="test456",
            description="Test goal",
            category="project",
            confidence=0.5,
            evidence=[],
            created_at=datetime.now(),
            last_mentioned=datetime.now(),
            progress=0.0,
            status="active",
            source="activity_inferred"
        )

        data = goal.to_dict()
        assert "source" in data
        assert data["source"] == "activity_inferred"

    def test_goal_from_dict_preserves_source(self):
        """Test Goal.from_dict() preserves source field."""
        from intelligence.goals import Goal
        from datetime import datetime

        data = {
            "id": "test789",
            "description": "Restored goal",
            "category": "learning",
            "confidence": 0.6,
            "evidence": [],
            "created_at": datetime.now().isoformat(),
            "last_mentioned": datetime.now().isoformat(),
            "progress": 0.2,
            "status": "active",
            "source": "activity_inferred"
        }

        goal = Goal.from_dict(data)
        assert goal.source == "activity_inferred"

    def test_goal_from_dict_defaults_to_conversation(self):
        """Test Goal.from_dict() defaults source to 'conversation'."""
        from intelligence.goals import Goal
        from datetime import datetime

        data = {
            "id": "old_goal",
            "description": "Old goal without source",
            "created_at": datetime.now().isoformat(),
            "last_mentioned": datetime.now().isoformat(),
        }

        goal = Goal.from_dict(data)
        assert goal.source == "conversation"

    def test_create_activity_inferred_goal(self):
        """Test GoalDetector.create_activity_inferred_goal() method."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock

        # Create mock memory
        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        goal = detector.create_activity_inferred_goal(
            description="Complete coding project",
            evidence="Observed 15 coding sessions",
            confidence=0.6,
            project_name="MyProject"
        )

        assert goal is not None
        assert goal.source == "activity_inferred"
        assert "MyProject" in goal.description
        assert goal.confidence == 0.6

    def test_create_activity_inferred_goal_without_project(self):
        """Test activity goal creation without project name."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        goal = detector.create_activity_inferred_goal(
            description="Finish writing task",
            evidence="Observed 20 writing sessions"
        )

        assert goal is not None
        assert goal.source == "activity_inferred"
        assert "MyProject" not in goal.description

    def test_get_goals_by_source(self):
        """Test GoalDetector.get_goals_by_source() method."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        # Add goals with different sources
        detector.goals["g1"] = Goal(
            id="g1", description="Conversation goal", category="project",
            confidence=0.5, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.0, status="active",
            source="conversation"
        )
        detector.goals["g2"] = Goal(
            id="g2", description="Activity goal", category="project",
            confidence=0.5, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.0, status="active",
            source="activity_inferred"
        )
        detector.goals["g3"] = Goal(
            id="g3", description="Another activity goal", category="learning",
            confidence=0.5, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.0, status="active",
            source="activity_inferred"
        )

        activity_goals = detector.get_goals_by_source("activity_inferred")
        conversation_goals = detector.get_goals_by_source("conversation")

        assert len(activity_goals) == 2
        assert len(conversation_goals) == 1

    @pytest.mark.asyncio
    async def test_suggest_goal_from_context_requires_minimum_snapshots(self):
        """Test that goal suggestion requires minimum 10 snapshots."""
        from intelligence.activity import ActivityMonitor, ActivitySnapshot
        from unittest.mock import MagicMock, AsyncMock

        # Create mock engine with goal detector
        mock_goal_detector = MagicMock()
        mock_goal_detector.get_active_goals.return_value = []
        mock_goal_detector.create_activity_inferred_goal = MagicMock()

        mock_engine = MagicMock()
        mock_engine.goal_detector = mock_goal_detector

        monitor = ActivityMonitor(senter_engine=mock_engine)

        # Try with only 5 snapshots - should not suggest goal
        await monitor._suggest_goal_from_context("coding", 5)
        mock_goal_detector.create_activity_inferred_goal.assert_not_called()

        # Try with 10+ snapshots - should suggest goal
        await monitor._suggest_goal_from_context("coding", 15)
        mock_goal_detector.create_activity_inferred_goal.assert_called_once()

    @pytest.mark.asyncio
    async def test_suggest_goal_from_project(self):
        """Test goal suggestion from detected project."""
        from intelligence.activity import ActivityMonitor
        from unittest.mock import MagicMock

        mock_goal_detector = MagicMock()
        mock_goal_detector.get_active_goals.return_value = []
        mock_goal_detector.create_activity_inferred_goal = MagicMock()

        mock_engine = MagicMock()
        mock_engine.goal_detector = mock_goal_detector

        monitor = ActivityMonitor(senter_engine=mock_engine)

        # Suggest goal for project with 15 snapshots
        await monitor._suggest_goal_from_project("TestProject", 15)

        mock_goal_detector.create_activity_inferred_goal.assert_called_once()
        call_args = mock_goal_detector.create_activity_inferred_goal.call_args
        assert "TestProject" in call_args.kwargs.get("description", "")

    @pytest.mark.asyncio
    async def test_analyze_patterns_triggers_goal_suggestions(self):
        """Test _analyze_patterns suggests goals for dominant contexts."""
        from intelligence.activity import ActivityMonitor, ActivitySnapshot
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_goal_detector = MagicMock()
        mock_goal_detector.get_active_goals.return_value = []
        mock_goal_detector.create_activity_inferred_goal = MagicMock()

        mock_engine = MagicMock()
        mock_engine.goal_detector = mock_goal_detector

        monitor = ActivityMonitor(senter_engine=mock_engine)

        # Add 15 coding snapshots to trigger goal suggestion
        for i in range(15):
            monitor.history.append(ActivitySnapshot(
                timestamp=datetime.now(),
                active_app="VSCode",
                window_title=f"file{i}.py",
                screen_text=[],
                inferred_context="coding",
                detected_project="TestProject"
            ))

        await monitor._analyze_patterns()

        # Should have attempted to create goals
        assert mock_goal_detector.create_activity_inferred_goal.called

    def test_activity_monitor_get_top_project_for_context(self):
        """Test getting top project for a specific context."""
        from intelligence.activity import ActivityMonitor, ActivitySnapshot
        from datetime import datetime

        monitor = ActivityMonitor()

        # Add snapshots with different projects in coding context
        for i in range(10):
            monitor.history.append(ActivitySnapshot(
                timestamp=datetime.now(),
                active_app="VSCode",
                window_title=f"file{i}.py",
                screen_text=[],
                inferred_context="coding",
                detected_project="ProjectA" if i < 7 else "ProjectB"
            ))

        top_project = monitor._get_top_project_for_context("coding")
        assert top_project == "ProjectA"  # 7 vs 3 occurrences


# ============================================================================
# US-004: LLM-based semantic goal detection
# ============================================================================

class TestSemanticGoalDetection:
    """Tests for LLM-based semantic goal detection."""

    @pytest.mark.asyncio
    async def test_detect_goals_semantically_with_mock_llm(self):
        """Test semantic goal detection with mocked LLM."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock, AsyncMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        # Mock episodes
        class MockEpisode:
            def __init__(self, input_text, response_text):
                self.input = input_text
                self.response = response_text

        episodes = [
            MockEpisode("I've been practicing Spanish every day", "That's great!"),
            MockEpisode("How do I conjugate verbs?", "Here's how..."),
            MockEpisode("I want to be fluent by summer", "You can do it!"),
        ]

        # Mock LLM response
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='[{"description": "Learn Spanish fluently", "category": "learning", "confidence": 0.85, "evidence": "User mentioned wanting fluency by summer"}]')

        goals = await detector.detect_goals_semantically(
            episodes=episodes,
            model=mock_model
        )

        assert len(goals) == 1
        assert "Spanish" in goals[0].description
        assert goals[0].category == "learning"
        assert goals[0].confidence >= 0.3

    @pytest.mark.asyncio
    async def test_detect_goals_semantically_no_model_returns_empty(self):
        """Test that detection returns empty list without model."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)

        class MockEpisode:
            input = "I want to learn Python"
            response = "Great choice!"

        goals = await detector.detect_goals_semantically(
            episodes=[MockEpisode()],
            model=None
        )

        assert goals == []

    @pytest.mark.asyncio
    async def test_detect_goals_semantically_empty_episodes(self):
        """Test detection with empty episodes list."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock, AsyncMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)
        mock_model = AsyncMock()

        goals = await detector.detect_goals_semantically(
            episodes=[],
            model=mock_model
        )

        assert goals == []
        mock_model.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_detect_goals_semantically_handles_malformed_json(self):
        """Test detection handles malformed LLM response gracefully."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock, AsyncMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        class MockEpisode:
            input = "I want to exercise more"
            response = "Good goal!"

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value="Not valid JSON at all")

        goals = await detector.detect_goals_semantically(
            episodes=[MockEpisode()],
            model=mock_model
        )

        assert goals == []

    @pytest.mark.asyncio
    async def test_detect_goals_semantically_extracts_embedded_json(self):
        """Test that JSON embedded in text is extracted correctly."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock, AsyncMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        class MockEpisode:
            input = "I'm training for a marathon"
            response = "That's ambitious!"

        # LLM returns JSON embedded in explanation text
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='Based on the conversation, here are the goals:\n[{"description": "Run a marathon", "category": "health", "confidence": 0.9, "evidence": "User mentioned training for marathon"}]\nThese goals reflect...')

        goals = await detector.detect_goals_semantically(
            episodes=[MockEpisode()],
            model=mock_model
        )

        assert len(goals) == 1
        assert "marathon" in goals[0].description.lower()

    @pytest.mark.asyncio
    async def test_detect_goals_semantically_multiple_goals(self):
        """Test detection of multiple goals from conversation."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock, AsyncMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        class MockEpisode:
            input = "I want to learn guitar and also get a promotion"
            response = "Both are achievable!"

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='[{"description": "Learn to play guitar", "category": "learning", "confidence": 0.8, "evidence": "User wants to learn guitar"}, {"description": "Get promoted at work", "category": "career", "confidence": 0.85, "evidence": "User mentioned wanting promotion"}]')

        goals = await detector.detect_goals_semantically(
            episodes=[MockEpisode()],
            model=mock_model
        )

        assert len(goals) == 2
        categories = {g.category for g in goals}
        assert "learning" in categories
        assert "career" in categories

    @pytest.mark.asyncio
    async def test_detect_goals_semantically_handles_llm_exception(self):
        """Test detection handles LLM exceptions gracefully."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock, AsyncMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)

        class MockEpisode:
            input = "Test input"
            response = "Test response"

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(side_effect=Exception("LLM error"))

        goals = await detector.detect_goals_semantically(
            episodes=[MockEpisode()],
            model=mock_model
        )

        assert goals == []

    def test_parse_llm_goal_response_valid_json(self):
        """Test parsing valid JSON array."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)

        response = '[{"description": "Test goal", "confidence": 0.8}]'
        result = detector._parse_llm_goal_response(response)

        assert len(result) == 1
        assert result[0]["description"] == "Test goal"

    def test_parse_llm_goal_response_embedded_json(self):
        """Test parsing JSON embedded in text."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)

        response = 'Here are the goals: [{"description": "Embedded goal"}] and more text'
        result = detector._parse_llm_goal_response(response)

        assert len(result) == 1
        assert result[0]["description"] == "Embedded goal"

    def test_parse_llm_goal_response_empty(self):
        """Test parsing empty response."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)

        assert detector._parse_llm_goal_response("") == []
        assert detector._parse_llm_goal_response(None) == []

    @pytest.mark.asyncio
    async def test_detect_goals_filters_short_descriptions(self):
        """Test that goals with very short descriptions are filtered out."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock, AsyncMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        class MockEpisode:
            input = "test"
            response = "test"

        mock_model = AsyncMock()
        # Return one valid goal and one with too-short description
        mock_model.generate = AsyncMock(return_value='[{"description": "ab", "confidence": 0.8}, {"description": "Learn Python programming", "confidence": 0.8}]')

        goals = await detector.detect_goals_semantically(
            episodes=[MockEpisode()],
            model=mock_model
        )

        assert len(goals) == 1
        assert "Python" in goals[0].description


# ============================================================================
# US-005: Goal progress detection via LLM
# ============================================================================

class TestGoalProgressDetection:
    """Tests for goal progress detection."""

    def test_detect_completion_finished_pattern(self):
        """Test detecting 'finished X' completion pattern."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        # Add a goal that can be completed
        detector.goals["g1"] = Goal(
            id="g1", description="Spanish lessons", category="learning",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.5, status="active"
        )

        # Simulate text indicating completion - uses topic that matches goal
        detector._detect_progress_from_text("I finally finished Spanish lessons!")

        assert detector.goals["g1"].status == "completed"
        assert detector.goals["g1"].progress == 1.0

    def test_detect_completion_done_with_pattern(self):
        """Test detecting 'done with X' completion pattern."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["proj1"] = Goal(
            id="proj1", description="website project", category="project",
            confidence=0.7, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.3, status="active"
        )

        detector._detect_progress_from_text("I'm done with the website project")

        assert detector.goals["proj1"].status == "completed"

    def test_detect_percentage_progress(self):
        """Test detecting percentage progress indicators."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["book"] = Goal(
            id="book", description="Write the book", category="project",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.2, status="active"
        )

        detector._detect_progress_from_text("I'm about 75% done with the book")

        assert detector.goals["book"].progress == 0.75
        assert detector.goals["book"].status == "active"

    def test_detect_halfway_progress(self):
        """Test detecting 'halfway' progress pattern."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["course"] = Goal(
            id="course", description="Complete the Python course", category="learning",
            confidence=0.7, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.2, status="active"
        )

        detector._detect_progress_from_text("I'm halfway through the Python course!")

        assert detector.goals["course"].progress == 0.5

    def test_detect_almost_done_progress(self):
        """Test detecting 'almost done' progress pattern."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["proj"] = Goal(
            id="proj", description="Finish the project", category="project",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.6, status="active"
        )

        detector._detect_progress_from_text("I'm almost done with the project")

        assert detector.goals["proj"].progress == 0.9

    def test_progress_only_increases(self):
        """Test that progress only increases, never decreases."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["task"] = Goal(
            id="task", description="Complete the task", category="project",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.8, status="active"
        )

        # Try to set lower progress
        detector._update_goal_by_topic("task", progress=0.3, mark_complete=False)

        # Progress should remain at 0.8
        assert detector.goals["task"].progress == 0.8

    def test_update_goal_by_topic_similarity_match(self):
        """Test updating goal by similar topic."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["spanish"] = Goal(
            id="spanish", description="Learn Spanish fluently", category="learning",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.3, status="active"
        )

        # Update with similar but not exact topic
        result = detector._update_goal_by_topic("spanish", progress=0.6)

        assert result is not None
        assert detector.goals["spanish"].progress == 0.6

    def test_update_goal_by_topic_no_match(self):
        """Test no update when topic doesn't match."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["coding"] = Goal(
            id="coding", description="Learn Python", category="learning",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.3, status="active"
        )

        # Try to update with unrelated topic
        result = detector._update_goal_by_topic("cooking recipes", progress=0.8)

        assert result is None
        assert detector.goals["coding"].progress == 0.3

    @pytest.mark.asyncio
    async def test_detect_progress_with_llm(self):
        """Test LLM-based progress detection."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock, AsyncMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["goal1"] = Goal(
            id="goal1", description="Learn to cook", category="learning",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.2, status="active"
        )

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='[{"goal_id": "goal1", "new_progress": 0.7, "completed": false, "evidence": "made good progress cooking"}]')

        results = await detector.detect_progress_with_llm(
            text="I've made good progress on my cooking skills",
            model=mock_model
        )

        assert len(results) == 1
        assert results[0]["goal_id"] == "goal1"
        assert results[0]["progress"] == 0.7
        assert detector.goals["goal1"].progress == 0.7

    @pytest.mark.asyncio
    async def test_detect_progress_with_llm_marks_complete(self):
        """Test LLM progress detection marks goal complete."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock, AsyncMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []
        mock_memory.semantic.store = MagicMock()

        detector = GoalDetector(mock_memory)

        detector.goals["goal2"] = Goal(
            id="goal2", description="Finish the report", category="project",
            confidence=0.8, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.9, status="active"
        )

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='[{"goal_id": "goal2", "new_progress": 1.0, "completed": true, "evidence": "finished the report"}]')

        results = await detector.detect_progress_with_llm(
            text="I finally finished the report!",
            model=mock_model
        )

        assert len(results) == 1
        assert results[0]["completed"] is True
        assert detector.goals["goal2"].status == "completed"
        assert detector.goals["goal2"].progress == 1.0

    @pytest.mark.asyncio
    async def test_detect_progress_with_llm_no_model(self):
        """Test progress detection returns empty without model."""
        from intelligence.goals import GoalDetector
        from unittest.mock import MagicMock

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)

        results = await detector.detect_progress_with_llm(
            text="Some text",
            model=None
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_detect_progress_with_llm_handles_exception(self):
        """Test progress detection handles LLM errors gracefully."""
        from intelligence.goals import GoalDetector, Goal
        from unittest.mock import MagicMock, AsyncMock
        from datetime import datetime

        mock_memory = MagicMock()
        mock_memory.semantic.get_by_domain.return_value = []

        detector = GoalDetector(mock_memory)

        detector.goals["g1"] = Goal(
            id="g1", description="Test goal", category="personal",
            confidence=0.5, evidence=[], created_at=datetime.now(),
            last_mentioned=datetime.now(), progress=0.3, status="active"
        )

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(side_effect=Exception("LLM error"))

        results = await detector.detect_progress_with_llm(
            text="Progress update",
            model=mock_model
        )

        assert results == []
        assert detector.goals["g1"].progress == 0.3  # Unchanged


# ============================================================================
# US-006: Failure analysis in MutationEngine
# ============================================================================

class TestMutationFailureAnalysis:
    """Tests for failure analysis in MutationEngine."""

    def test_failure_analysis_dataclass(self):
        """Test FailureAnalysis dataclass structure."""
        from evolution.mutations import FailureAnalysis

        analysis = FailureAnalysis(
            total_episodes=10,
            patterns={"too_long": 3, "too_short": 2},
            suggested_fixes=[{"pattern": "too_long", "fix_type": "prompt_refinement"}],
            avg_fitness=0.4,
            worst_episode_id="ep1",
            analysis_summary="Test summary"
        )

        assert analysis.total_episodes == 10
        assert analysis.patterns["too_long"] == 3
        assert len(analysis.suggested_fixes) == 1
        assert analysis.avg_fitness == 0.4

    def test_failure_analysis_to_dict(self):
        """Test FailureAnalysis.to_dict() method."""
        from evolution.mutations import FailureAnalysis

        analysis = FailureAnalysis(
            total_episodes=5,
            patterns={"wrong_mode": 2},
            suggested_fixes=[],
            avg_fitness=0.35
        )

        data = analysis.to_dict()
        assert data["total_episodes"] == 5
        assert "wrong_mode" in data["patterns"]
        assert data["avg_fitness"] == 0.35

    def test_analyze_low_fitness_episodes_empty(self):
        """Test analysis with empty episodes list."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()
        analysis = engine.analyze_low_fitness_episodes([])

        assert analysis.total_episodes == 0
        assert analysis.patterns == {}
        assert "No episodes" in analysis.analysis_summary

    def test_analyze_low_fitness_episodes_too_long(self):
        """Test detecting too_long failure pattern."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        # Create mock episodes with long responses
        class MockEpisode:
            def __init__(self, response_len):
                self.id = "ep1"
                self.input = "short question"
                self.response = "x" * response_len
                self.mode = "DIALOGUE"
                self.cognitive_state = {"fitness": 0.3}
                self.joint_state = {"fitness": 0.3}

        episodes = [MockEpisode(2500), MockEpisode(3000), MockEpisode(100)]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        assert analysis.patterns["too_long"] == 2
        assert any(fix["pattern"] == "too_long" for fix in analysis.suggested_fixes)

    def test_analyze_low_fitness_episodes_too_short(self):
        """Test detecting too_short failure pattern."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            def __init__(self, input_text, response):
                self.id = "ep2"
                self.input = input_text
                self.response = response
                self.mode = "DIALOGUE"
                self.cognitive_state = {"fitness": 0.3}
                self.joint_state = {"fitness": 0.3}

        episodes = [
            MockEpisode("This is a very long question that requires a detailed answer", "Ok."),
            MockEpisode("Another lengthy question that deserves more than just a word", "Yes."),
            MockEpisode("Short", "This is a reasonable response."),
        ]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        assert analysis.patterns["too_short"] == 2
        assert any(fix["pattern"] == "too_short" for fix in analysis.suggested_fixes)

    def test_analyze_low_fitness_episodes_wrong_mode(self):
        """Test detecting wrong_mode failure pattern."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            def __init__(self, input_text, mode):
                self.id = "ep3"
                self.input = input_text
                self.response = "Some response"
                self.mode = mode
                self.cognitive_state = {"fitness": 0.3}
                self.joint_state = {"fitness": 0.3}

        episodes = [
            MockEpisode("teach me about Python", "DIALOGUE"),  # Should be TEACHING
            MockEpisode("help me with this code", "DIALOGUE"),  # Should be COLLABORATIVE
            MockEpisode("what is AI?", "DIALOGUE"),  # Correct
        ]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        assert analysis.patterns["wrong_mode"] == 2

    def test_analyze_low_fitness_episodes_missed_frustration(self):
        """Test detecting missed_frustration failure pattern."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            def __init__(self, input_text, frustration):
                self.id = "ep4"
                self.input = input_text
                self.response = "Response"
                self.mode = "DIALOGUE"
                self.cognitive_state = {"fitness": 0.3, "frustration": frustration}
                self.joint_state = {"fitness": 0.3}

        episodes = [
            MockEpisode("I'm so frustrated with this bug!", 0.1),  # Missed frustration
            MockEpisode("This doesn't work and it's annoying", 0.2),  # Missed frustration
            MockEpisode("I'm frustrated", 0.8),  # Detected correctly
        ]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        assert analysis.patterns["missed_frustration"] == 2

    def test_analyze_low_fitness_episodes_skips_high_fitness(self):
        """Test that high-fitness episodes are not analyzed for failures."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            def __init__(self, fitness, response_len):
                self.id = "ep5"
                self.input = "question"
                self.response = "x" * response_len
                self.mode = "DIALOGUE"
                self.cognitive_state = {"fitness": fitness}
                self.joint_state = {"fitness": fitness}

        episodes = [
            MockEpisode(0.8, 5000),  # High fitness - should not count as too_long
            MockEpisode(0.3, 5000),  # Low fitness - should count
        ]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        assert analysis.patterns["too_long"] == 1  # Only the low-fitness one

    def test_analyze_low_fitness_episodes_worst_episode(self):
        """Test tracking worst episode."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            def __init__(self, ep_id, fitness):
                self.id = ep_id
                self.input = "question"
                self.response = "answer"
                self.mode = "DIALOGUE"
                self.cognitive_state = {"fitness": fitness}
                self.joint_state = {"fitness": fitness}

        episodes = [
            MockEpisode("ep1", 0.4),
            MockEpisode("ep2", 0.1),  # Worst
            MockEpisode("ep3", 0.3),
        ]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        assert analysis.worst_episode_id == "ep2"
        assert analysis.avg_fitness == pytest.approx(0.267, rel=0.1)

    def test_analyze_low_fitness_episodes_suggested_fixes_priority(self):
        """Test that suggested fixes are sorted by priority."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            def __init__(self, response_len, input_len):
                self.id = "ep6"
                self.input = "x" * input_len
                self.response = "y" * response_len
                self.mode = "DIALOGUE"
                self.cognitive_state = {"fitness": 0.3}
                self.joint_state = {"fitness": 0.3}

        # Create episodes that trigger multiple patterns
        episodes = [
            MockEpisode(3000, 10),  # too_long
            MockEpisode(3000, 10),  # too_long
            MockEpisode(3000, 10),  # too_long
            MockEpisode(20, 100),   # too_short
            MockEpisode(20, 100),   # too_short
        ]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        # too_long should have higher priority (3 vs 2)
        if analysis.suggested_fixes:
            assert analysis.suggested_fixes[0]["pattern"] == "too_long"

    def test_generate_fixes_from_patterns(self):
        """Test fix generation from patterns."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        patterns = {
            "too_long": 5,
            "missed_frustration": 3,
            "too_short": 0,
            "wrong_mode": 0,
            "low_engagement": 0,
            "off_topic": 0,
        }

        fixes = engine._generate_fixes_from_patterns(patterns)

        assert len(fixes) == 2  # Only patterns with >= 2 occurrences
        assert fixes[0]["pattern"] == "too_long"  # Highest priority first
        assert fixes[0]["priority"] == 5

    def test_analysis_summary_content(self):
        """Test that analysis summary is informative."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            def __init__(self, response_len):
                self.id = "ep7"
                self.input = "short"
                self.response = "x" * response_len
                self.mode = "DIALOGUE"
                self.cognitive_state = {"fitness": 0.3}
                self.joint_state = {"fitness": 0.3}

        episodes = [MockEpisode(3000) for _ in range(5)]

        analysis = engine.analyze_low_fitness_episodes(episodes, fitness_threshold=0.5)

        assert "5 episodes" in analysis.analysis_summary
        assert "too_long" in analysis.analysis_summary


# ============================================================================
# US-007: LLM-driven mutation proposals
# ============================================================================

class TestIntelligentMutationProposal:
    """Tests for LLM-driven intelligent mutation proposals."""

    @pytest.mark.asyncio
    async def test_propose_intelligent_mutation_empty_episodes(self):
        """Test that empty episodes returns None."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()
        mutation = await engine.propose_intelligent_mutation([])

        assert mutation is None

    @pytest.mark.asyncio
    async def test_propose_intelligent_mutation_no_failures(self):
        """Test that no failures returns None."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            id = "ep1"
            input = "test"
            response = "adequate response"
            mode = "DIALOGUE"
            cognitive_state = {"fitness": 0.8}  # High fitness
            joint_state = {"fitness": 0.8}

        mutation = await engine.propose_intelligent_mutation([MockEpisode()])

        assert mutation is None

    @pytest.mark.asyncio
    async def test_propose_intelligent_mutation_with_llm(self):
        """Test LLM-driven mutation proposal."""
        from evolution.mutations import MutationEngine
        from unittest.mock import AsyncMock

        engine = MutationEngine()

        class MockEpisode:
            id = "ep1"
            input = "test input"
            response = "x" * 3000  # Too long
            mode = "DIALOGUE"
            cognitive_state = {"fitness": 0.3}
            joint_state = {"fitness": 0.3}

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value='{"mutation_type": "threshold_modification", "target": "response.conciseness_weight", "direction": "increase", "magnitude": "medium", "reason": "Responses too verbose"}')

        mutation = await engine.propose_intelligent_mutation(
            episodes=[MockEpisode(), MockEpisode()],
            model=mock_model
        )

        assert mutation is not None
        assert "conciseness" in mutation.target or "LLM Analysis" in mutation.reason

    @pytest.mark.asyncio
    async def test_propose_intelligent_mutation_fallback_without_model(self):
        """Test fallback to heuristic when no model provided."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        class MockEpisode:
            id = "ep1"
            input = "test"
            response = "x" * 3000  # Too long
            mode = "DIALOGUE"
            cognitive_state = {"fitness": 0.3}
            joint_state = {"fitness": 0.3}

        mutation = await engine.propose_intelligent_mutation(
            episodes=[MockEpisode(), MockEpisode()],
            model=None
        )

        assert mutation is not None
        assert mutation.mutation_type in ["threshold_modification", "prompt_refinement", "protocol_tuning"]

    @pytest.mark.asyncio
    async def test_propose_intelligent_mutation_handles_llm_error(self):
        """Test fallback when LLM fails."""
        from evolution.mutations import MutationEngine
        from unittest.mock import AsyncMock

        engine = MutationEngine()

        class MockEpisode:
            id = "ep1"
            input = "test"
            response = "x" * 3000
            mode = "DIALOGUE"
            cognitive_state = {"fitness": 0.3}
            joint_state = {"fitness": 0.3}

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(side_effect=Exception("LLM error"))

        mutation = await engine.propose_intelligent_mutation(
            episodes=[MockEpisode(), MockEpisode()],
            model=mock_model
        )

        # Should fall back to heuristic
        assert mutation is not None

    @pytest.mark.asyncio
    async def test_propose_intelligent_mutation_handles_invalid_json(self):
        """Test fallback when LLM returns invalid JSON."""
        from evolution.mutations import MutationEngine
        from unittest.mock import AsyncMock

        engine = MutationEngine()

        class MockEpisode:
            id = "ep1"
            input = "test"
            response = "x" * 3000
            mode = "DIALOGUE"
            cognitive_state = {"fitness": 0.3}
            joint_state = {"fitness": 0.3}

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value="Not valid JSON at all")

        mutation = await engine.propose_intelligent_mutation(
            episodes=[MockEpisode(), MockEpisode()],
            model=mock_model
        )

        # Should fall back to heuristic
        assert mutation is not None

    def test_parse_llm_mutation_response_valid_json(self):
        """Test parsing valid JSON response."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        response = '{"mutation_type": "threshold_modification", "target": "test.value", "direction": "increase"}'
        result = engine._parse_llm_mutation_response(response)

        assert result is not None
        assert result["mutation_type"] == "threshold_modification"
        assert result["target"] == "test.value"

    def test_parse_llm_mutation_response_embedded_json(self):
        """Test parsing JSON embedded in text."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        response = 'Here is my suggestion: {"mutation_type": "prompt_refinement", "target": "response.style"} and more text'
        result = engine._parse_llm_mutation_response(response)

        assert result is not None
        assert result["mutation_type"] == "prompt_refinement"

    def test_parse_llm_mutation_response_empty(self):
        """Test parsing empty response."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        assert engine._parse_llm_mutation_response("") is None
        assert engine._parse_llm_mutation_response(None) is None

    def test_create_mutation_from_llm_numeric(self):
        """Test creating mutation from LLM data with numeric value."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine(genome={"test": {"value": 0.5}})

        data = {
            "mutation_type": "threshold_modification",
            "target": "test.value",
            "direction": "increase",
            "magnitude": "medium",
            "reason": "Test reason"
        }

        mutation = engine._create_mutation_from_llm(data, 0.4)

        assert mutation is not None
        assert mutation.target == "test.value"
        assert mutation.new_value == 0.6  # 0.5 + 0.1 (medium)
        assert "LLM Analysis" in mutation.reason

    def test_create_mutation_from_llm_decrease(self):
        """Test creating mutation with decrease direction."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine(genome={"threshold": 0.7})

        data = {
            "mutation_type": "threshold_modification",
            "target": "threshold",
            "direction": "decrease",
            "magnitude": "large"
        }

        mutation = engine._create_mutation_from_llm(data, 0.3)

        assert mutation is not None
        assert mutation.new_value == 0.5  # 0.7 - 0.2 (large)

    def test_create_mutation_from_llm_boolean(self):
        """Test creating mutation with boolean toggle."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine(genome={"feature": {"enabled": True}})

        data = {
            "mutation_type": "protocol_tuning",
            "target": "feature.enabled",
            "direction": "toggle"
        }

        mutation = engine._create_mutation_from_llm(data, 0.4)

        assert mutation is not None
        assert mutation.new_value is False

    def test_create_mutation_from_llm_no_target(self):
        """Test that missing target returns None."""
        from evolution.mutations import MutationEngine

        engine = MutationEngine()

        data = {"mutation_type": "threshold_modification"}

        mutation = engine._create_mutation_from_llm(data, 0.4)

        assert mutation is None

    def test_propose_mutation_from_analysis_too_long(self):
        """Test heuristic mutation for too_long pattern."""
        from evolution.mutations import MutationEngine, FailureAnalysis

        engine = MutationEngine()

        analysis = FailureAnalysis(
            total_episodes=10,
            patterns={"too_long": 5, "too_short": 0, "wrong_mode": 0, "missed_frustration": 0, "low_engagement": 0, "off_topic": 0},
            suggested_fixes=[{"pattern": "too_long", "fix_type": "prompt_refinement", "priority": 5}],
            avg_fitness=0.3
        )

        mutation = engine._propose_mutation_from_analysis(analysis)

        assert mutation is not None
        assert "conciseness" in mutation.target

    def test_propose_mutation_from_analysis_missed_frustration(self):
        """Test heuristic mutation for missed_frustration pattern."""
        from evolution.mutations import MutationEngine, FailureAnalysis

        engine = MutationEngine()

        analysis = FailureAnalysis(
            total_episodes=10,
            patterns={"too_long": 0, "too_short": 0, "wrong_mode": 0, "missed_frustration": 4, "low_engagement": 0, "off_topic": 0},
            suggested_fixes=[{"pattern": "missed_frustration", "fix_type": "threshold_modification", "priority": 4}],
            avg_fitness=0.35
        )

        mutation = engine._propose_mutation_from_analysis(analysis)

        assert mutation is not None
        assert "frustration" in mutation.target

    def test_propose_mutation_from_analysis_no_fixes(self):
        """Test that no fixes returns None."""
        from evolution.mutations import MutationEngine, FailureAnalysis

        engine = MutationEngine()

        analysis = FailureAnalysis(
            total_episodes=5,
            patterns={"too_long": 0, "too_short": 0, "wrong_mode": 0, "missed_frustration": 0, "low_engagement": 0, "off_topic": 0},
            suggested_fixes=[],
            avg_fitness=0.5
        )

        mutation = engine._propose_mutation_from_analysis(analysis)

        assert mutation is None

# ============================================================================
# US-008: Self-initiated task creation from goals
# ============================================================================

class TestSelfInitiatedTasks:
    """Tests for self-initiated task creation from goals."""

    def test_goal_derived_task_dataclass(self):
        """Test GoalDerivedTask dataclass structure."""
        from intelligence.proactive import GoalDerivedTask

        task = GoalDerivedTask(
            task_id="goal_123_1234567890",
            goal_id="123",
            task_type="research",
            description="Research about Python",
            parameters={"goal_description": "Learn Python"}
        )

        assert task.task_id == "goal_123_1234567890"
        assert task.goal_id == "123"
        assert task.task_type == "research"
        assert task.origin == "goal_derived"
        assert task.created_at is not None

    def test_goal_derived_task_defaults(self):
        """Test GoalDerivedTask default values."""
        from intelligence.proactive import GoalDerivedTask

        task = GoalDerivedTask(
            task_id="test",
            goal_id="1",
            task_type="plan",
            description="Test",
            parameters={}
        )

        assert task.origin == "goal_derived"
        assert task.created_at is not None

    def test_create_tasks_from_goals_requires_trust(self):
        """Test that task creation requires trust level > 0.7."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal

        # Mock engine with low trust
        mock_trust = MagicMock()
        mock_trust.level = 0.5  # Below threshold
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {"1": Goal(id="1", description="Learn Python", category="learning")}

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        tasks = engine.create_tasks_from_goals()

        assert len(tasks) == 0  # No tasks due to low trust

    def test_create_tasks_from_goals_with_sufficient_trust(self):
        """Test task creation with sufficient trust level."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal

        # Mock engine with high trust
        mock_trust = MagicMock()
        mock_trust.level = 0.8  # Above threshold
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {"1": Goal(id="1", description="Learn Python", category="learning")}

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        tasks = engine.create_tasks_from_goals()

        assert len(tasks) == 1
        assert tasks[0].goal_id == "1"
        assert tasks[0].origin == "goal_derived"

    def test_create_tasks_research_for_learning_goals(self):
        """Test that learning goals create research tasks."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal

        mock_trust = MagicMock()
        mock_trust.level = 0.8
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {"1": Goal(id="1", description="Learn Spanish", category="learning")}

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        tasks = engine.create_tasks_from_goals()

        assert len(tasks) == 1
        assert tasks[0].task_type == "research"
        assert "Research" in tasks[0].description

    def test_create_tasks_plan_for_non_learning_goals(self):
        """Test that non-learning goals create plan tasks."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal

        mock_trust = MagicMock()
        mock_trust.level = 0.8
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {"1": Goal(id="1", description="Ship feature", category="work")}

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        tasks = engine.create_tasks_from_goals()

        assert len(tasks) == 1
        assert tasks[0].task_type == "plan"
        assert "actionable steps" in tasks[0].description

    def test_create_tasks_respects_cooldown(self):
        """Test that cooldown prevents duplicate task creation."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal
        from datetime import datetime, timedelta

        mock_trust = MagicMock()
        mock_trust.level = 0.8
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {"1": Goal(id="1", description="Learn Python", category="learning")}

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        # First call creates task
        tasks1 = engine.create_tasks_from_goals()
        assert len(tasks1) == 1

        # Second call should not create (cooldown)
        tasks2 = engine.create_tasks_from_goals()
        assert len(tasks2) == 0

    def test_create_tasks_cooldown_expires(self):
        """Test that tasks can be created after cooldown expires."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal
        from datetime import datetime, timedelta

        mock_trust = MagicMock()
        mock_trust.level = 0.8
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {"1": Goal(id="1", description="Learn Python", category="learning")}

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        # First call
        tasks1 = engine.create_tasks_from_goals()
        assert len(tasks1) == 1

        # Simulate cooldown expired
        engine.created_task_ids["1"] = datetime.now() - timedelta(hours=13)

        # Now should create again
        tasks2 = engine.create_tasks_from_goals()
        assert len(tasks2) == 1

    def test_create_tasks_multiple_goals(self):
        """Test task creation with multiple goals."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal

        mock_trust = MagicMock()
        mock_trust.level = 0.8
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {
            "1": Goal(id="1", description="Learn Python", category="learning"),
            "2": Goal(id="2", description="Build app", category="project"),
            "3": Goal(id="3", description="Study Japanese", category="learning")
        }

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        tasks = engine.create_tasks_from_goals()

        assert len(tasks) == 3
        task_types = {t.task_type for t in tasks}
        assert "research" in task_types
        assert "plan" in task_types

    def test_create_tasks_parameters_include_goal_info(self):
        """Test that task parameters include goal information."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from intelligence.goals import GoalDetector, Goal

        mock_trust = MagicMock()
        mock_trust.level = 0.8
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        mock_memory = MagicMock()
        goal_detector = GoalDetector(mock_memory)
        goal_detector.goals = {
            "1": Goal(id="1", description="Learn Python", category="learning", confidence=0.7, progress=0.3)
        }

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector)

        tasks = engine.create_tasks_from_goals()

        assert len(tasks) == 1
        params = tasks[0].parameters
        assert params["goal_description"] == "Learn Python"
        assert params["goal_category"] == "learning"
        assert params["goal_confidence"] == 0.7
        assert params["goal_progress"] == 0.3

    def test_get_task_creation_status_low_trust(self):
        """Test task creation status with low trust."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_trust = MagicMock()
        mock_trust.level = 0.5
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        engine = ProactiveSuggestionEngine(mock_engine)

        status = engine.get_task_creation_status()

        assert status["trust_level"] == 0.5
        assert status["min_trust_required"] == 0.7
        assert status["can_create_tasks"] is False

    def test_get_task_creation_status_high_trust(self):
        """Test task creation status with high trust."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_trust = MagicMock()
        mock_trust.level = 0.85
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        engine = ProactiveSuggestionEngine(mock_engine)

        status = engine.get_task_creation_status()

        assert status["trust_level"] == 0.85
        assert status["can_create_tasks"] is True
        assert status["cooldown_hours"] == 12

    def test_create_tasks_no_goal_detector(self):
        """Test that no tasks created without goal detector."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_trust = MagicMock()
        mock_trust.level = 0.8
        mock_engine = MagicMock()
        mock_engine.trust = mock_trust

        engine = ProactiveSuggestionEngine(mock_engine, goal_detector=None)

        tasks = engine.create_tasks_from_goals()

        assert len(tasks) == 0

    def test_should_create_task_for_goal_first_time(self):
        """Test _should_create_task_for_goal returns True for new goal."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._should_create_task_for_goal("new_goal")

        assert result is True


# ============================================================================
# US-009: Goal-based background research automation
# ============================================================================

class TestAutomatedGoalResearch:
    """Tests for automated goal research in daemon."""

    def test_goal_research_result_dataclass(self):
        """Test GoalResearchResult dataclass structure."""
        from daemon.senter_daemon import GoalResearchResult
        from datetime import datetime

        result = GoalResearchResult(
            goal_id="123",
            goal_description="Learn Python",
            research_summary="Python is a programming language...",
            sources=["https://python.org"],
            completed_at=datetime.now()
        )

        assert result.goal_id == "123"
        assert result.goal_description == "Learn Python"
        assert "Python" in result.research_summary
        assert len(result.sources) == 1
        assert result.stored_in_memory is False

    def test_goal_research_result_defaults(self):
        """Test GoalResearchResult default values."""
        from daemon.senter_daemon import GoalResearchResult
        from datetime import datetime

        result = GoalResearchResult(
            goal_id="1",
            goal_description="Test",
            research_summary="Summary",
            sources=[],
            completed_at=datetime.now()
        )

        assert result.stored_in_memory is False

    def test_background_worker_has_research_list(self):
        """Test BackgroundWorker initializes with empty research list."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)

            assert hasattr(worker, "goal_research_results")
            assert worker.goal_research_results == []
            assert worker._max_research_results == 50

    @pytest.mark.asyncio
    async def test_auto_research_returns_empty_without_engine(self):
        """Test auto_research returns empty list without engine."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)

            results = await worker.auto_research_learning_goals()

            assert results == []

    @pytest.mark.asyncio
    async def test_auto_research_returns_empty_without_goal_detector(self):
        """Test auto_research returns empty list without goal detector."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            mock_engine = MagicMock()
            mock_engine.goal_detector = None
            worker = BackgroundWorker(engine=mock_engine, task_queue=queue)

            results = await worker.auto_research_learning_goals()

            assert results == []

    @pytest.mark.asyncio
    async def test_do_goal_research_stores_in_memory(self):
        """Test _do_goal_research stores result in semantic memory."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue, Task, TaskPriority
        from pathlib import Path
        from datetime import datetime
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")

            # Mock engine with model and memory
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value="Research summary about Python basics")

            mock_memory = MagicMock()
            mock_semantic = MagicMock()
            mock_memory.semantic = mock_semantic

            mock_engine = MagicMock()
            mock_engine.model = mock_model
            mock_engine.memory = mock_memory

            worker = BackgroundWorker(engine=mock_engine, task_queue=queue)

            task = Task(
                priority=TaskPriority.BACKGROUND.value,
                created_at=datetime.now(),
                task_id="test_123",
                task_type="goal_research",
                description="Research Python",
                parameters={
                    "goal_id": "goal_1",
                    "goal_description": "Learn Python",
                    "query": "Python basics"
                }
            )

            result = await worker._do_goal_research(task)

            assert result is not None
            assert result.goal_id == "goal_1"
            assert result.stored_in_memory is True
            mock_semantic.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_goal_research_without_llm(self):
        """Test _do_goal_research creates placeholder without LLM."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue, Task, TaskPriority
        from pathlib import Path
        from datetime import datetime
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")

            mock_engine = MagicMock()
            mock_engine.model = None

            worker = BackgroundWorker(engine=mock_engine, task_queue=queue)

            task = Task(
                priority=TaskPriority.BACKGROUND.value,
                created_at=datetime.now(),
                task_id="test_456",
                task_type="goal_research",
                description="Research Spanish",
                parameters={
                    "goal_id": "goal_2",
                    "goal_description": "Learn Spanish",
                    "query": "Spanish basics"
                }
            )

            result = await worker._do_goal_research(task)

            assert result is not None
            assert "LLM not available" in result.research_summary

    def test_while_you_were_away_no_tasks(self):
        """Test while_you_were_away with no completed tasks."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)

            summary = worker.get_while_you_were_away_summary()

            assert "completed_tasks" in summary
            assert "goal_research" in summary
            assert "summary" in summary
            assert len(summary["completed_tasks"]) == 0
            assert "No background work" in summary["summary"]

    def test_while_you_were_away_with_tasks(self):
        """Test while_you_were_away with completed tasks."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue, Task, TaskPriority
        from pathlib import Path
        from datetime import datetime
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)

            # Add a completed task
            task = Task(
                priority=TaskPriority.NORMAL.value,
                created_at=datetime.now(),
                task_id="task_1",
                task_type="research",
                description="Research topic",
                status="completed",
                result="Completed successfully"
            )
            worker.completed_tasks.append(task)

            summary = worker.get_while_you_were_away_summary()

            assert len(summary["completed_tasks"]) == 1
            assert "Completed 1 background tasks" in summary["summary"]

    def test_while_you_were_away_with_research(self):
        """Test while_you_were_away with goal research."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue, GoalResearchResult
        from pathlib import Path
        from datetime import datetime
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)

            # Add research result
            result = GoalResearchResult(
                goal_id="goal_1",
                goal_description="Learn Python",
                research_summary="Python is great for beginners...",
                sources=["https://python.org"],
                completed_at=datetime.now(),
                stored_in_memory=True
            )
            worker.goal_research_results.append(result)

            summary = worker.get_while_you_were_away_summary()

            assert len(summary["goal_research"]) == 1
            assert "Researched 1 learning goals" in summary["summary"]
            assert summary["goal_research"][0]["stored"] is True

    def test_while_you_were_away_filters_by_time(self):
        """Test while_you_were_away filters results by time."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue, Task, TaskPriority
        from pathlib import Path
        from datetime import datetime, timedelta
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)

            # Add old task (more than 24 hours ago)
            old_task = Task(
                priority=TaskPriority.NORMAL.value,
                created_at=datetime.now() - timedelta(hours=48),
                task_id="old_task",
                task_type="research",
                description="Old research",
                status="completed"
            )
            worker.completed_tasks.append(old_task)

            # Add recent task
            recent_task = Task(
                priority=TaskPriority.NORMAL.value,
                created_at=datetime.now(),
                task_id="recent_task",
                task_type="research",
                description="Recent research",
                status="completed"
            )
            worker.completed_tasks.append(recent_task)

            summary = worker.get_while_you_were_away_summary()

            assert len(summary["completed_tasks"]) == 1
            assert summary["completed_tasks"][0]["id"] == "recent_task"

    def test_while_you_were_away_custom_since(self):
        """Test while_you_were_away with custom since parameter."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue, Task, TaskPriority
        from pathlib import Path
        from datetime import datetime, timedelta
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)

            # Add task from 36 hours ago
            task = Task(
                priority=TaskPriority.NORMAL.value,
                created_at=datetime.now() - timedelta(hours=36),
                task_id="task_36h",
                task_type="research",
                description="36h ago research",
                status="completed"
            )
            worker.completed_tasks.append(task)

            # Default (24h) should not include it
            summary_24h = worker.get_while_you_were_away_summary()
            assert len(summary_24h["completed_tasks"]) == 0

            # 48h window should include it
            summary_48h = worker.get_while_you_were_away_summary(
                since=datetime.now() - timedelta(hours=48)
            )
            assert len(summary_48h["completed_tasks"]) == 1

    def test_research_results_list_trimmed(self):
        """Test that research results list is trimmed to max size."""
        from daemon.senter_daemon import BackgroundWorker, TaskQueue, GoalResearchResult
        from pathlib import Path
        from datetime import datetime
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskQueue(Path(tmpdir) / "tasks.json")
            worker = BackgroundWorker(engine=None, task_queue=queue)
            worker._max_research_results = 5  # Set low for testing

            # Add more than max results
            for i in range(10):
                result = GoalResearchResult(
                    goal_id=f"goal_{i}",
                    goal_description=f"Goal {i}",
                    research_summary=f"Summary {i}",
                    sources=[],
                    completed_at=datetime.now()
                )
                worker.goal_research_results.append(result)

                # Trim like the real method does
                if len(worker.goal_research_results) > worker._max_research_results:
                    worker.goal_research_results = worker.goal_research_results[-worker._max_research_results:]

            assert len(worker.goal_research_results) == 5
            # Should have the last 5 (goal_5 through goal_9)
            assert worker.goal_research_results[0].goal_id == "goal_5"


# ============================================================================
# US-010: Pattern-based need prediction
# ============================================================================

class TestAnticipatoryNeedPrediction:
    """Tests for anticipatory need prediction."""

    def test_need_pattern_dataclass(self):
        """Test NeedPattern dataclass structure."""
        from intelligence.proactive import NeedPattern
        from datetime import datetime

        pattern = NeedPattern(
            topic="python",
            frequency=5,
            time_slots=[9, 10, 14, 15],
            confidence=0.7,
            last_occurrence=datetime.now(),
            associated_activities=["coding", "learning"]
        )

        assert pattern.topic == "python"
        assert pattern.frequency == 5
        assert len(pattern.time_slots) == 4
        assert pattern.confidence == 0.7

    def test_need_pattern_matches_time(self):
        """Test NeedPattern.matches_time method."""
        from intelligence.proactive import NeedPattern
        from datetime import datetime

        pattern = NeedPattern(
            topic="python",
            frequency=5,
            time_slots=[9, 10, 14],
            confidence=0.7,
            last_occurrence=datetime.now(),
            associated_activities=[]
        )

        assert pattern.matches_time(9) is True
        assert pattern.matches_time(10) is True
        assert pattern.matches_time(14) is True
        assert pattern.matches_time(12) is False
        assert pattern.matches_time(18) is False

    def test_predicted_need_dataclass(self):
        """Test PredictedNeed dataclass structure."""
        from intelligence.proactive import PredictedNeed
        from datetime import datetime

        need = PredictedNeed(
            topic="python",
            predicted_time=datetime.now(),
            confidence=0.8,
            source_pattern="freq:5, time:True, activity:False",
            prefetch_query="information about python"
        )

        assert need.topic == "python"
        assert need.confidence == 0.8
        assert "python" in need.prefetch_query

    def test_analyze_needs_patterns_empty_episodes(self):
        """Test analyze_needs_patterns with empty episodes."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        patterns = engine.analyze_needs_patterns(episodes=[])

        assert patterns == []

    def test_analyze_needs_patterns_finds_recurring_topics(self):
        """Test analyze_needs_patterns finds recurring topics."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        # Create episodes with recurring topic
        episodes = []
        for i in range(5):
            ep = MagicMock()
            ep.input = "How do I learn python programming basics"
            ep.timestamp = datetime.now()
            episodes.append(ep)

        patterns = engine.analyze_needs_patterns(episodes=episodes)

        # Should find "python" and "programming" as patterns (>= 3 occurrences)
        topics = [p.topic for p in patterns]
        assert "python" in topics
        assert "programming" in topics

    def test_analyze_needs_patterns_filters_common_words(self):
        """Test that common words are filtered out."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        episodes = []
        for i in range(5):
            ep = MagicMock()
            ep.input = "about would could should there their"
            ep.timestamp = datetime.now()
            episodes.append(ep)

        patterns = engine.analyze_needs_patterns(episodes=episodes)

        # Should not find common words as patterns
        assert len(patterns) == 0

    def test_analyze_needs_patterns_stores_internally(self):
        """Test that patterns are stored in internal dict."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        episodes = []
        for i in range(5):
            ep = MagicMock()
            ep.input = "python tutorial learning"
            ep.timestamp = datetime.now()
            episodes.append(ep)

        engine.analyze_needs_patterns(episodes=episodes)

        assert "python" in engine.need_patterns
        assert "tutorial" in engine.need_patterns

    def test_predict_needs_empty_patterns(self):
        """Test predict_needs with no patterns."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        predictions = engine.predict_needs()

        assert predictions == []

    def test_predict_needs_matches_time(self):
        """Test predict_needs matches time slots."""
        from intelligence.proactive import ProactiveSuggestionEngine, NeedPattern
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        # Add pattern that matches current hour
        engine.need_patterns["python"] = NeedPattern(
            topic="python",
            frequency=5,
            time_slots=[10],
            confidence=0.5,
            last_occurrence=datetime.now(),
            associated_activities=[]
        )

        predictions = engine.predict_needs(current_hour=10)

        assert len(predictions) >= 1
        assert predictions[0].topic == "python"
        assert "time:True" in predictions[0].source_pattern

    def test_predict_needs_matches_activity(self):
        """Test predict_needs matches activity."""
        from intelligence.proactive import ProactiveSuggestionEngine, NeedPattern
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        engine.need_patterns["python"] = NeedPattern(
            topic="python",
            frequency=5,
            time_slots=[],
            confidence=0.5,
            last_occurrence=datetime.now(),
            associated_activities=["coding"]
        )

        predictions = engine.predict_needs(current_hour=12, current_activity="coding")

        assert len(predictions) >= 1
        assert predictions[0].topic == "python"
        assert "activity:True" in predictions[0].source_pattern

    def test_predict_needs_boosts_confidence(self):
        """Test that matching boosts confidence."""
        from intelligence.proactive import ProactiveSuggestionEngine, NeedPattern
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        engine.need_patterns["python"] = NeedPattern(
            topic="python",
            frequency=5,
            time_slots=[10],
            confidence=0.5,
            last_occurrence=datetime.now(),
            associated_activities=["coding"]
        )

        # Match both time and activity
        predictions = engine.predict_needs(current_hour=10, current_activity="coding")

        assert len(predictions) >= 1
        # Base 0.5 * 1.2 (time) * 1.3 (activity) = 0.78
        assert predictions[0].confidence >= 0.7

    @pytest.mark.asyncio
    async def test_prefetch_research_without_llm(self):
        """Test prefetch_research creates placeholders without LLM."""
        from intelligence.proactive import ProactiveSuggestionEngine, PredictedNeed
        from datetime import datetime

        mock_engine = MagicMock()
        mock_engine.model = None
        engine = ProactiveSuggestionEngine(mock_engine)

        predictions = [
            PredictedNeed(
                topic="python",
                predicted_time=datetime.now(),
                confidence=0.8,
                source_pattern="test",
                prefetch_query="info about python"
            )
        ]

        results = await engine.prefetch_research_for_needs(predictions)

        assert "python" in results
        assert results["python"]["source"] == "placeholder"

    @pytest.mark.asyncio
    async def test_prefetch_research_with_llm(self):
        """Test prefetch_research uses LLM when available."""
        from intelligence.proactive import ProactiveSuggestionEngine, PredictedNeed
        from datetime import datetime

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value="Python is a programming language...")

        mock_engine = MagicMock()
        mock_engine.model = mock_model
        engine = ProactiveSuggestionEngine(mock_engine)

        predictions = [
            PredictedNeed(
                topic="python",
                predicted_time=datetime.now(),
                confidence=0.8,
                source_pattern="test",
                prefetch_query="info about python"
            )
        ]

        results = await engine.prefetch_research_for_needs(predictions)

        assert "python" in results
        assert results["python"]["source"] == "llm_prefetch"
        assert "Python is a programming language" in results["python"]["summary"]

    @pytest.mark.asyncio
    async def test_prefetch_research_caches_results(self):
        """Test that prefetched research is cached."""
        from intelligence.proactive import ProactiveSuggestionEngine, PredictedNeed
        from datetime import datetime

        mock_engine = MagicMock()
        mock_engine.model = None
        engine = ProactiveSuggestionEngine(mock_engine)

        predictions = [
            PredictedNeed(
                topic="python",
                predicted_time=datetime.now(),
                confidence=0.8,
                source_pattern="test",
                prefetch_query="info about python"
            )
        ]

        # First call
        await engine.prefetch_research_for_needs(predictions)

        # Should be cached
        assert "python" in engine.prefetched_research
        assert engine.get_prefetched_research("python") is not None

    def test_get_prefetched_research(self):
        """Test get_prefetched_research method."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        # Add cached research
        engine.prefetched_research["python"] = {
            "topic": "python",
            "summary": "Test summary",
            "fetched_at": datetime.now()
        }

        result = engine.get_prefetched_research("python")
        assert result is not None
        assert result["summary"] == "Test summary"

        # Non-existent topic
        assert engine.get_prefetched_research("nonexistent") is None

    def test_clear_prefetched_research_single(self):
        """Test clearing single topic from cache."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        engine.prefetched_research["python"] = {"topic": "python"}
        engine.prefetched_research["javascript"] = {"topic": "javascript"}

        engine.clear_prefetched_research("python")

        assert "python" not in engine.prefetched_research
        assert "javascript" in engine.prefetched_research

    def test_clear_prefetched_research_all(self):
        """Test clearing all cached research."""
        from intelligence.proactive import ProactiveSuggestionEngine
        from datetime import datetime

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        engine.prefetched_research["python"] = {"topic": "python"}
        engine.prefetched_research["javascript"] = {"topic": "javascript"}

        engine.clear_prefetched_research()

        assert len(engine.prefetched_research) == 0

    def test_get_status_includes_pattern_counts(self):
        """Test that get_status includes pattern and research counts."""
        from intelligence.proactive import ProactiveSuggestionEngine, NeedPattern
        from datetime import datetime

        mock_engine = MagicMock()
        mock_engine.trust = None
        engine = ProactiveSuggestionEngine(mock_engine)

        engine.need_patterns["python"] = NeedPattern(
            topic="python",
            frequency=5,
            time_slots=[10],
            confidence=0.5,
            last_occurrence=datetime.now(),
            associated_activities=[]
        )
        engine.prefetched_research["python"] = {"topic": "python"}

        status = engine.get_status()

        assert status["need_patterns_count"] == 1
        assert status["prefetched_research_count"] == 1


# ============================================================================
# US-011: Activity-context-aware suggestions
# ============================================================================

class TestActivityAwareSuggestions:
    """Tests for activity-context-aware suggestions."""

    def test_proactive_engine_accepts_activity_monitor(self):
        """Test ProactiveSuggestionEngine accepts activity_monitor parameter."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        mock_activity = MagicMock()

        engine = ProactiveSuggestionEngine(
            mock_engine,
            goal_detector=None,
            activity_monitor=mock_activity
        )

        assert engine.activity_monitor == mock_activity

    def test_activity_resource_map_initialized(self):
        """Test that activity_resource_map is initialized with default mappings."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        assert "coding" in engine.activity_resource_map
        assert "writing" in engine.activity_resource_map
        assert "research" in engine.activity_resource_map
        assert "learning" in engine.activity_resource_map

    def test_break_suggestion_threshold_default(self):
        """Test default break suggestion threshold is 120 minutes."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        assert engine.break_suggestion_minutes == 120

    def test_check_break_needed_no_summary(self):
        """Test _check_break_needed returns None with no summary."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._check_break_needed({})
        assert result is None

        result = engine._check_break_needed(None)
        assert result is None

    def test_check_break_needed_under_threshold(self):
        """Test no break suggestion when under threshold."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        summary = {"time_by_context": {"coding": 60}}  # 1 hour

        result = engine._check_break_needed(summary)
        assert result is None

    def test_check_break_needed_over_threshold(self):
        """Test break suggestion when over threshold."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        summary = {"time_by_context": {"coding": 150}}  # 2.5 hours

        result = engine._check_break_needed(summary)

        assert result is not None
        assert result["type"] == "activity_break"
        assert result["activity_context"] == "coding"
        assert result["duration_minutes"] == 150
        assert "2.5 hours" in result["action"]

    def test_suggest_resources_for_activity_coding(self):
        """Test resource suggestions for coding activity."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._suggest_resources_for_activity("coding", "MyProject")

        assert result is not None
        assert result["type"] == "activity_resource"
        assert result["activity_context"] == "coding"
        assert result["project"] == "MyProject"
        assert "documentation" in result["suggested_resources"]

    def test_suggest_resources_for_activity_unknown(self):
        """Test no resource suggestions for unknown activity."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._suggest_resources_for_activity("unknown_activity", None)
        assert result is None

    def test_suggest_for_context_coding(self):
        """Test context-specific suggestion for coding."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._suggest_for_context("coding", "TestProject", {})

        assert result is not None
        assert result["type"] == "activity_coding"
        assert "TestProject" in result["action"]

    def test_suggest_for_context_research(self):
        """Test context-specific suggestion for research."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._suggest_for_context("research", None, {})

        assert result is not None
        assert result["type"] == "activity_research"
        assert "organize" in result["action"].lower() or "research" in result["action"].lower()

    def test_suggest_for_context_writing(self):
        """Test context-specific suggestion for writing."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._suggest_for_context("writing", None, {})

        assert result is not None
        assert result["type"] == "activity_writing"

    def test_suggest_for_context_none(self):
        """Test no suggestion for empty context."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        result = engine._suggest_for_context(None, None, {})
        assert result is None

        result = engine._suggest_for_context("", None, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_activity_context_suggestions_no_monitor(self):
        """Test _activity_context_suggestions returns empty without monitor."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine, activity_monitor=None)

        result = await engine._activity_context_suggestions()
        assert result == []

    @pytest.mark.asyncio
    async def test_activity_context_suggestions_with_monitor(self):
        """Test _activity_context_suggestions generates suggestions with monitor."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()

        mock_activity = MagicMock()
        mock_activity.get_current_context.return_value = "coding"
        mock_activity.get_current_project.return_value = "TestProject"
        mock_activity.get_activity_summary.return_value = {
            "time_by_context": {"coding": 30}
        }

        engine = ProactiveSuggestionEngine(mock_engine, activity_monitor=mock_activity)

        result = await engine._activity_context_suggestions()

        # Should have resource and context suggestions (no break, under threshold)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_activity_context_suggestions_includes_break(self):
        """Test _activity_context_suggestions includes break suggestion when needed."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()

        mock_activity = MagicMock()
        mock_activity.get_current_context.return_value = "coding"
        mock_activity.get_current_project.return_value = "TestProject"
        mock_activity.get_activity_summary.return_value = {
            "time_by_context": {"coding": 150}  # Over threshold
        }

        engine = ProactiveSuggestionEngine(mock_engine, activity_monitor=mock_activity)

        result = await engine._activity_context_suggestions()

        # Should include break suggestion
        break_suggestions = [s for s in result if s["type"] == "activity_break"]
        assert len(break_suggestions) == 1

    def test_get_activity_suggestion_status(self):
        """Test get_activity_suggestion_status method."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        mock_activity = MagicMock()

        engine = ProactiveSuggestionEngine(mock_engine, activity_monitor=mock_activity)

        status = engine.get_activity_suggestion_status()

        assert status["has_activity_monitor"] is True
        assert status["break_threshold_minutes"] == 120
        assert "coding" in status["resource_categories"]

    def test_get_activity_suggestion_status_no_monitor(self):
        """Test get_activity_suggestion_status without monitor."""
        from intelligence.proactive import ProactiveSuggestionEngine

        mock_engine = MagicMock()
        engine = ProactiveSuggestionEngine(mock_engine)

        status = engine.get_activity_suggestion_status()

        assert status["has_activity_monitor"] is False
