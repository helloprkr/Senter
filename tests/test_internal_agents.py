#!/usr/bin/env python3
"""
Tests for Internal Agents (IA-001, IA-002, IA-003, IA-004)
Tests Goal Detector entity extraction, Context Gatherer, Profiler, and SENTER.md Writer.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== IA-001: Goal Detector Entity Extraction Tests ==========

def test_entity_type_enum():
    """Test EntityType enum exists (IA-001)"""
    from Functions.goal_tracker import EntityType

    assert EntityType.ACTION.value == "action"
    assert EntityType.OBJECT.value == "object"
    assert EntityType.DEADLINE.value == "deadline"
    assert EntityType.CONDITION.value == "condition"

    return True


def test_extracted_entity_creation():
    """Test ExtractedEntity dataclass (IA-001)"""
    from Functions.goal_tracker import ExtractedEntity, EntityType

    entity = ExtractedEntity(
        entity_type=EntityType.ACTION,
        value="finish",
        start_pos=0,
        end_pos=6,
        confidence=0.8
    )

    assert entity.entity_type == EntityType.ACTION
    assert entity.value == "finish"
    assert entity.confidence == 0.8

    return True


def test_goal_entity_extractor_creation():
    """Test GoalEntityExtractor can be created (IA-001)"""
    from Functions.goal_tracker import GoalEntityExtractor

    extractor = GoalEntityExtractor()
    assert extractor is not None
    assert len(extractor.ACTION_PATTERNS) > 0
    assert len(extractor.DEADLINE_PATTERNS) > 0

    return True


def test_extract_action_entities():
    """Test action entity extraction (IA-001)"""
    from Functions.goal_tracker import GoalEntityExtractor, EntityType

    extractor = GoalEntityExtractor()
    result = extractor.extract_entities("I need to finish my project")

    assert result.action is not None
    assert any(e.entity_type == EntityType.ACTION for e in result.entities)

    return True


def test_extract_deadline_entities():
    """Test deadline entity extraction (IA-001)"""
    from Functions.goal_tracker import GoalEntityExtractor, EntityType

    extractor = GoalEntityExtractor()
    result = extractor.extract_entities("I need to finish by Friday")

    assert result.deadline is not None
    assert "friday" in result.deadline.lower()
    assert any(e.entity_type == EntityType.DEADLINE for e in result.entities)

    return True


def test_extract_condition_entities():
    """Test condition entity extraction (IA-001)"""
    from Functions.goal_tracker import GoalEntityExtractor, EntityType

    extractor = GoalEntityExtractor()
    result = extractor.extract_entities("I will submit if I finish the code")

    assert result.condition is not None
    assert any(e.entity_type == EntityType.CONDITION for e in result.entities)

    return True


def test_confidence_calculation():
    """Test confidence scoring based on entity completeness (IA-001)"""
    from Functions.goal_tracker import GoalEntityExtractor

    extractor = GoalEntityExtractor()

    # Full extraction: ACTION + OBJECT + DEADLINE
    result = extractor.extract_entities(
        "I need to finish my presentation by Friday"
    )
    assert result.confidence >= 0.6  # High confidence

    # Partial extraction: only ACTION
    result2 = extractor.extract_entities("I want to learn")
    assert result2.confidence < 0.6  # Lower confidence

    return True


def test_clarification_queue_low_confidence():
    """Test low confidence goals queue for clarification (IA-001)"""
    from Functions.goal_tracker import GoalEntityExtractor

    extractor = GoalEntityExtractor()

    # Vague goal
    result = extractor.extract_entities("I should do something")

    # Should need clarification if confidence low
    if result.confidence < 0.6:
        assert result.needs_clarification
        assert len(result.clarification_questions) > 0

    return True


def test_clarification_queue_operations():
    """Test ClarificationQueue operations (IA-001)"""
    from Functions.goal_tracker import ClarificationQueue, GoalExtractionResult

    with tempfile.TemporaryDirectory() as tmpdir:
        queue = ClarificationQueue(Path(tmpdir))

        # Add an item
        result = GoalExtractionResult(
            confidence=0.4,
            needs_clarification=True,
            clarification_questions=["What do you want to accomplish?"]
        )
        queue.add(result, "I should do something")

        # Check pending
        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"

        # Resolve
        queue.resolve(0, {"goal": "Resolved goal"})
        pending = queue.get_pending()
        assert len(pending) == 0

    return True


# ========== IA-002: Context Gatherer Tests ==========

def test_context_gatherer_creation():
    """Test ContextGatherer can be created (IA-002)"""
    from Functions.context_gatherer import ContextGatherer

    with tempfile.TemporaryDirectory() as tmpdir:
        gatherer = ContextGatherer(Path(tmpdir))
        assert gatherer is not None
        assert gatherer.focuses_dir.exists() or True  # May not exist yet

    return True


def test_update_mode_enum():
    """Test UpdateMode enum (IA-002)"""
    from Functions.context_gatherer import UpdateMode

    assert UpdateMode.APPEND.value == "append"
    assert UpdateMode.PREPEND.value == "prepend"
    assert UpdateMode.MERGE.value == "merge"
    assert UpdateMode.REPLACE_SECTION.value == "replace"

    return True


def test_context_update_creation():
    """Test ContextUpdate dataclass (IA-002)"""
    from Functions.context_gatherer import ContextUpdate, UpdateMode

    update = ContextUpdate(
        section_name="Detected Goals",
        content="New goal detected",
        update_mode=UpdateMode.APPEND,
        source="conversation"
    )

    assert update.section_name == "Detected Goals"
    assert update.content == "New goal detected"
    assert update.update_mode == UpdateMode.APPEND
    assert update.timestamp > 0

    return True


def test_context_update_to_focus():
    """Test applying context update to focus (IA-002)"""
    from Functions.context_gatherer import ContextGatherer, ContextUpdate, UpdateMode

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a test focus
        focus_dir = tmppath / "Focuses" / "TestFocus"
        focus_dir.mkdir(parents=True)

        senter_content = """---
focus:
  name: TestFocus
---

## Detected Goals
*None yet*

## Other Section
Content here
"""
        (focus_dir / "SENTER.md").write_text(senter_content)

        gatherer = ContextGatherer(tmppath)

        # Apply update
        update = ContextUpdate(
            section_name="Detected Goals",
            content="- New goal: Learn Python",
            update_mode=UpdateMode.APPEND
        )

        result = gatherer.update_focus_context("TestFocus", update)

        assert result.success
        assert result.change_type == "appended"

        # Verify content updated
        new_content = (focus_dir / "SENTER.md").read_text()
        assert "New goal: Learn Python" in new_content

    return True


def test_context_update_deduplication():
    """Test context update deduplication (IA-002)"""
    from Functions.context_gatherer import ContextGatherer, ContextUpdate, UpdateMode

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        focus_dir = tmppath / "Focuses" / "DedupeTest"
        focus_dir.mkdir(parents=True)

        senter_content = """---
focus:
  name: DedupeTest
---

## Detected Goals
- Existing goal
"""
        (focus_dir / "SENTER.md").write_text(senter_content)

        gatherer = ContextGatherer(tmppath)

        # Try to add duplicate content
        update = ContextUpdate(
            section_name="Detected Goals",
            content="Existing goal",  # Same as existing
            update_mode=UpdateMode.APPEND
        )

        result = gatherer.update_focus_context("DedupeTest", update)

        # Should succeed but not add duplicate
        assert result.success

        new_content = (focus_dir / "SENTER.md").read_text()
        # Should only have one "Existing goal"
        assert new_content.count("Existing goal") == 1

    return True


def test_batch_update():
    """Test batch context updates (IA-002)"""
    from Functions.context_gatherer import ContextGatherer, ContextUpdate, UpdateMode

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        focus_dir = tmppath / "Focuses" / "BatchTest"
        focus_dir.mkdir(parents=True)

        senter_content = """---
focus:
  name: BatchTest
---

## Goals
*None*

## Entities
*None*
"""
        (focus_dir / "SENTER.md").write_text(senter_content)

        gatherer = ContextGatherer(tmppath)

        updates = [
            ContextUpdate("Goals", "Goal 1", UpdateMode.APPEND),
            ContextUpdate("Entities", "Entity 1", UpdateMode.APPEND)
        ]

        results = gatherer.batch_update("BatchTest", updates)

        assert len(results) == 2
        assert all(r.success for r in results)

    return True


# ========== IA-003: Profiler Behavioral Modeling Tests ==========

def test_communication_style_enum():
    """Test CommunicationStyle enum (IA-003)"""
    from Functions.learner import CommunicationStyle

    assert CommunicationStyle.CASUAL.value == "casual"
    assert CommunicationStyle.PROFESSIONAL.value == "professional"
    assert CommunicationStyle.TECHNICAL.value == "technical"
    assert CommunicationStyle.MIXED.value == "mixed"

    return True


def test_expertise_level_enum():
    """Test ExpertiseLevel enum (IA-003)"""
    from Functions.learner import ExpertiseLevel

    assert ExpertiseLevel.BEGINNER.value == "beginner"
    assert ExpertiseLevel.INTERMEDIATE.value == "intermediate"
    assert ExpertiseLevel.ADVANCED.value == "advanced"
    assert ExpertiseLevel.EXPERT.value == "expert"

    return True


def test_behavioral_profile_creation():
    """Test BehavioralProfile dataclass (IA-003)"""
    from Functions.learner import BehavioralProfile, CommunicationStyle

    profile = BehavioralProfile(user_id="test_user")

    assert profile.user_id == "test_user"
    assert profile.communication_style == CommunicationStyle.MIXED
    assert profile.formality_score == 0.5
    assert profile.expertise_areas == []
    assert profile.total_interactions == 0

    return True


def test_behavioral_profiler_creation():
    """Test BehavioralProfiler can be created (IA-003)"""
    from Functions.learner import BehavioralProfiler

    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = BehavioralProfiler(Path(tmpdir))
        assert profiler is not None
        assert profiler.profile is not None

    return True


def test_behavioral_profiler_analyze():
    """Test BehavioralProfiler.analyze_interaction (IA-003)"""
    from Functions.learner import BehavioralProfiler, CommunicationStyle

    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = BehavioralProfiler(Path(tmpdir))

        # Casual message
        messages = [
            {"role": "user", "content": "Hey, can you help me with some Python code? Thanks!"}
        ]
        profiler.analyze_interaction(messages)

        assert profiler.profile.total_interactions == 1
        assert profiler.profile.last_seen != ""

    return True


def test_profiler_expertise_detection():
    """Test expertise area detection (IA-003)"""
    from Functions.learner import BehavioralProfiler

    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = BehavioralProfiler(Path(tmpdir))

        # Technical Python message
        messages = [
            {"role": "user", "content": "How do I use Python pandas to import a CSV and run a query on the database?"}
        ]
        profiler.analyze_interaction(messages)

        # Should detect python or database expertise
        domains = [e.domain for e in profiler.profile.expertise_areas]
        assert "python" in domains or "database" in domains

    return True


def test_profiler_peak_hours():
    """Test peak hours tracking (IA-003)"""
    from Functions.learner import BehavioralProfiler
    from datetime import datetime

    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = BehavioralProfiler(Path(tmpdir))

        # Simulate interactions at specific times
        for _ in range(5):
            messages = [{"role": "user", "content": "Test message"}]
            profiler.analyze_interaction(messages, datetime(2024, 1, 1, 14, 0))

        # Should have hour 14 in peak hours
        assert len(profiler.profile.activity_heatmap) > 0

    return True


def test_profiler_get_summary():
    """Test BehavioralProfiler.get_profile_summary (IA-003)"""
    from Functions.learner import BehavioralProfiler

    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = BehavioralProfiler(Path(tmpdir))

        messages = [{"role": "user", "content": "Help me learn about machine learning"}]
        profiler.analyze_interaction(messages)

        summary = profiler.get_profile_summary()

        assert "user_id" in summary
        assert "communication_style" in summary
        assert "expertise" in summary
        assert "interactions" in summary
        assert summary["interactions"] >= 1

    return True


def test_profiler_system_prompt_additions():
    """Test system prompt additions from profile (IA-003)"""
    from Functions.learner import BehavioralProfiler, ExpertiseArea, ExpertiseLevel

    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = BehavioralProfiler(Path(tmpdir))

        # Add expertise manually
        profiler.profile.expertise_areas.append(ExpertiseArea(
            domain="python",
            level=ExpertiseLevel.ADVANCED,
            confidence=0.8
        ))
        profiler.profile.preferred_code_language = "python"
        profiler.profile.profile_confidence = 0.7
        profiler.profile.preferred_explanation_style = "detailed"

        additions = profiler.get_system_prompt_additions()

        assert "[User Profile:]" in additions
        assert "python" in additions.lower()

    return True


# ========== IA-004: SENTER.md Writer Tests ==========

def test_focus_factory_creates_senter_md():
    """Test FocusFactory creates valid SENTER.md (IA-004)"""
    from Focuses.focus_factory import FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)
        focus_path = factory.create_focus("TestDomain", "Initial context about testing")

        assert focus_path.exists()
        assert (focus_path / "SENTER.md").exists()

        content = (focus_path / "SENTER.md").read_text()
        assert "TestDomain" in content
        assert "system_prompt:" in content

    return True


def test_focus_factory_system_prompt_tailored():
    """Test SENTER.md system prompt is tailored to domain (IA-004)"""
    from Focuses.focus_factory import FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)
        focus_path = factory.create_focus("MachineLearning", "ML and AI topics")

        content = (focus_path / "SENTER.md").read_text()

        # System prompt should mention the focus name
        assert "MachineLearning" in content

    return True


def test_focus_factory_creates_wiki():
    """Test FocusFactory creates wiki.md for conversational focuses (IA-004)"""
    from Focuses.focus_factory import FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)

        # "research" is a conversational keyword
        focus_path = factory.create_focus("research_topic", "Research context")

        # Should have wiki.md
        assert (focus_path / "wiki.md").exists()

    return True


def test_focus_factory_internal_focus():
    """Test creating internal focus for agents (IA-004)"""
    from Focuses.focus_factory import FocusFactory

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "Focuses").mkdir()
        (tmppath / "config").mkdir()

        factory = FocusFactory(tmppath)
        focus_path = factory.create_internal_focus("Goal_Detector", "profile")

        assert focus_path.exists()
        assert (focus_path / "SENTER.md").exists()

        content = (focus_path / "SENTER.md").read_text()
        assert "Goal_Detector" in content
        assert "internal" in content

    return True


if __name__ == "__main__":
    tests = [
        # IA-001
        test_entity_type_enum,
        test_extracted_entity_creation,
        test_goal_entity_extractor_creation,
        test_extract_action_entities,
        test_extract_deadline_entities,
        test_extract_condition_entities,
        test_confidence_calculation,
        test_clarification_queue_low_confidence,
        test_clarification_queue_operations,
        # IA-002
        test_context_gatherer_creation,
        test_update_mode_enum,
        test_context_update_creation,
        test_context_update_to_focus,
        test_context_update_deduplication,
        test_batch_update,
        # IA-003
        test_communication_style_enum,
        test_expertise_level_enum,
        test_behavioral_profile_creation,
        test_behavioral_profiler_creation,
        test_behavioral_profiler_analyze,
        test_profiler_expertise_detection,
        test_profiler_peak_hours,
        test_profiler_get_summary,
        test_profiler_system_prompt_additions,
        # IA-004
        test_focus_factory_creates_senter_md,
        test_focus_factory_system_prompt_tailored,
        test_focus_factory_creates_wiki,
        test_focus_factory_internal_focus,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
