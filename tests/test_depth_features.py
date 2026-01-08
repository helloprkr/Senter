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