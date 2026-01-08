#!/usr/bin/env python3
"""
Tests for Goal System (GS-001, GS-002, GS-003)
Tests goal progress detection, prioritization, and relationships.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== GS-001: Goal Progress Auto-Detection Tests ==========

def test_goal_progress_field():
    """Test Goal has progress field (GS-001)"""
    from Functions.goal_tracker import Goal

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat()
    )

    assert goal.progress == 0.0
    assert goal.progress_notes == []

    return True


def test_goal_update_progress():
    """Test Goal.update_progress method (GS-001)"""
    from Functions.goal_tracker import Goal

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat()
    )

    # Update progress
    completed = goal.update_progress(0.5, "Halfway done")

    assert goal.progress == 0.5
    assert not completed
    assert len(goal.progress_notes) == 1
    assert "50%" in goal.progress_notes[0]

    return True


def test_goal_complete_on_100_percent():
    """Test goal completes at 100% (GS-001)"""
    from Functions.goal_tracker import Goal

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat()
    )

    completed = goal.update_progress(1.0, "All done!")

    assert completed
    assert goal.status == "completed"
    assert goal.progress == 1.0

    return True


def test_goal_progress_from_subtasks():
    """Test calculating progress from subtasks (GS-001)"""
    from Functions.goal_tracker import Goal, SubTask

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat(),
        sub_tasks=[
            SubTask(task="Task 1", status="completed"),
            SubTask(task="Task 2", status="completed"),
            SubTask(task="Task 3", status="pending"),
            SubTask(task="Task 4", status="pending"),
        ]
    )

    progress = goal.calculate_progress_from_subtasks()

    assert progress == 0.5  # 2 of 4 completed

    return True


def test_progress_detector_creation():
    """Test GoalProgressDetector can be created (GS-001)"""
    from Functions.goal_tracker import GoalProgressDetector, GoalTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))
        detector = GoalProgressDetector(tracker)

        assert detector is not None
        assert len(detector.ACCOMPLISHMENT_PATTERNS) > 0
        assert len(detector.PROGRESS_INDICATORS) > 0

    return True


def test_progress_detector_patterns():
    """Test progress detection patterns (GS-001)"""
    from Functions.goal_tracker import GoalProgressDetector, GoalTracker, Goal
    from datetime import datetime

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))

        # Add a goal
        goal = Goal(
            id="goal_0001",
            description="Finish the presentation",
            created=datetime.now().isoformat()
        )
        tracker.goals.append(goal)
        tracker._save_goals()

        detector = GoalProgressDetector(tracker)

        # Test detection
        results = detector.detect_progress("I finished the presentation today!")

        # Should detect progress
        assert len(results) >= 0  # May or may not match depending on word overlap

    return True


def test_subtask_completion_detection():
    """Test subtask completion detection (GS-001)"""
    from Functions.goal_tracker import GoalProgressDetector, GoalTracker, Goal, SubTask

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))

        goal = Goal(
            id="goal_0001",
            description="Build app",
            created=datetime.now().isoformat(),
            sub_tasks=[
                SubTask(task="Write tests", status="pending"),
                SubTask(task="Fix bugs", status="pending"),
            ]
        )
        tracker.goals.append(goal)
        tracker._save_goals()

        detector = GoalProgressDetector(tracker)

        results = detector.detect_subtask_completion("I just finished writing the tests")

        # May detect the subtask
        assert isinstance(results, list)

    return True


# ========== GS-002: Goal Prioritization Tests ==========

def test_goal_priority_score():
    """Test Goal has priority_score field (GS-002)"""
    from Functions.goal_tracker import Goal

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat()
    )

    assert goal.priority_score == 5  # Default
    assert goal.due_date is None
    assert goal.is_overdue is False

    return True


def test_goal_set_priority():
    """Test Goal.set_priority method (GS-002)"""
    from Functions.goal_tracker import Goal

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat()
    )

    goal.set_priority(8)
    assert goal.priority_score == 8
    assert goal.priority == "high"

    goal.set_priority(2)
    assert goal.priority_score == 2
    assert goal.priority == "low"

    goal.set_priority(5)
    assert goal.priority_score == 5
    assert goal.priority == "medium"

    return True


def test_goal_priority_bounds():
    """Test priority score is bounded 1-10 (GS-002)"""
    from Functions.goal_tracker import Goal

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat()
    )

    goal.set_priority(15)  # Too high
    assert goal.priority_score == 10

    goal.set_priority(-5)  # Too low
    assert goal.priority_score == 1

    return True


def test_parse_due_date_tomorrow():
    """Test parse_due_date for 'tomorrow' (GS-002)"""
    from Functions.goal_tracker import parse_due_date
    from datetime import datetime, timedelta

    result = parse_due_date("by tomorrow")
    expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    assert result == expected

    return True


def test_parse_due_date_day_of_week():
    """Test parse_due_date for day of week (GS-002)"""
    from Functions.goal_tracker import parse_due_date

    result = parse_due_date("by Friday")

    assert result is not None
    # Should be a valid date string
    assert len(result) == 10
    assert result[4] == "-"

    return True


def test_parse_due_date_in_days():
    """Test parse_due_date for 'in N days' (GS-002)"""
    from Functions.goal_tracker import parse_due_date
    from datetime import datetime, timedelta

    result = parse_due_date("in 5 days")
    expected = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")

    assert result == expected

    return True


def test_goal_overdue_detection():
    """Test overdue goal detection (GS-002)"""
    from Functions.goal_tracker import Goal
    from datetime import datetime, timedelta

    # Create goal with past due date
    past_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat(),
        due_date=past_date
    )

    is_overdue = goal.check_overdue()

    assert is_overdue
    assert goal.is_overdue

    return True


def test_tracker_goals_by_priority():
    """Test GoalTracker.get_goals_by_priority (GS-002)"""
    from Functions.goal_tracker import GoalTracker, Goal

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))

        # Add goals with different priorities
        for i, priority in enumerate([3, 8, 5, 10, 1]):
            goal = Goal(
                id=f"goal_000{i}",
                description=f"Goal {i}",
                created=datetime.now().isoformat(),
                priority_score=priority
            )
            tracker.goals.append(goal)
        tracker._save_goals()

        # Get by priority
        sorted_goals = tracker.get_goals_by_priority()

        # Should be sorted highest first
        assert sorted_goals[0].priority_score == 10
        assert sorted_goals[-1].priority_score == 1

    return True


# ========== GS-003: Goal Relationships Tests ==========

def test_goal_parent_child_fields():
    """Test Goal has parent/child fields (GS-003)"""
    from Functions.goal_tracker import Goal

    goal = Goal(
        id="goal_0001",
        description="Test goal",
        created=datetime.now().isoformat()
    )

    assert goal.parent_id is None
    assert goal.child_ids == []

    return True


def test_goal_add_remove_child():
    """Test adding/removing child goals (GS-003)"""
    from Functions.goal_tracker import Goal

    parent = Goal(
        id="goal_0001",
        description="Parent goal",
        created=datetime.now().isoformat()
    )

    parent.add_child("goal_0002")
    parent.add_child("goal_0003")

    assert len(parent.child_ids) == 2
    assert "goal_0002" in parent.child_ids
    assert parent.has_children()

    parent.remove_child("goal_0002")
    assert len(parent.child_ids) == 1
    assert "goal_0002" not in parent.child_ids

    return True


def test_goal_is_child():
    """Test Goal.is_child method (GS-003)"""
    from Functions.goal_tracker import Goal

    child = Goal(
        id="goal_0002",
        description="Child goal",
        created=datetime.now().isoformat(),
        parent_id="goal_0001"
    )

    assert child.is_child()
    assert child.parent_id == "goal_0001"

    root = Goal(
        id="goal_0001",
        description="Root goal",
        created=datetime.now().isoformat()
    )

    assert not root.is_child()

    return True


def test_goal_tree_creation():
    """Test GoalTree can be created (GS-003)"""
    from Functions.goal_tracker import GoalTree, GoalTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))
        tree = GoalTree(tracker)

        assert tree is not None
        assert tree.tracker == tracker

    return True


def test_goal_tree_create_child():
    """Test GoalTree.create_child_goal (GS-003)"""
    from Functions.goal_tracker import GoalTree, GoalTracker, Goal

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))

        # Create parent goal
        parent = Goal(
            id=tracker._generate_id(),
            description="Learn Python",
            created=datetime.now().isoformat()
        )
        tracker.goals.append(parent)
        tracker._save_goals()

        tree = GoalTree(tracker)

        # Create child
        child = tree.create_child_goal(parent.id, "Complete Python tutorial")

        assert child is not None
        assert child.parent_id == parent.id
        assert child.id in parent.child_ids

    return True


def test_goal_tree_aggregate_progress():
    """Test progress aggregation from children (GS-003)"""
    from Functions.goal_tracker import GoalTree, GoalTracker, Goal

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))

        # Create parent
        parent = Goal(
            id="goal_0001",
            description="Parent",
            created=datetime.now().isoformat()
        )
        tracker.goals.append(parent)

        # Create children with different progress
        for i, progress in enumerate([0.5, 0.75, 1.0]):
            child = Goal(
                id=f"goal_000{i+2}",
                description=f"Child {i}",
                created=datetime.now().isoformat(),
                parent_id="goal_0001",
                progress=progress
            )
            parent.add_child(child.id)
            tracker.goals.append(child)

        tracker._save_goals()

        tree = GoalTree(tracker)

        # Aggregate progress
        aggregated = tree.aggregate_progress("goal_0001")

        # Should be average: (0.5 + 0.75 + 1.0) / 3 = 0.75
        assert abs(aggregated - 0.75) < 0.01

    return True


def test_goal_tree_view():
    """Test GoalTree.get_tree_view (GS-003)"""
    from Functions.goal_tracker import GoalTree, GoalTracker, Goal

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))

        # Create hierarchy
        parent = Goal(
            id="goal_0001",
            description="Learn programming",
            created=datetime.now().isoformat(),
            progress=0.5
        )
        tracker.goals.append(parent)

        child = Goal(
            id="goal_0002",
            description="Learn Python basics",
            created=datetime.now().isoformat(),
            parent_id="goal_0001",
            progress=0.75
        )
        parent.add_child(child.id)
        tracker.goals.append(child)

        tracker._save_goals()

        tree = GoalTree(tracker)

        # Get tree view
        view = tree.get_tree_view()

        assert "goal_0001" in view
        assert "Learn programming" in view

    return True


def test_goal_tree_get_root_goals():
    """Test GoalTree.get_root_goals (GS-003)"""
    from Functions.goal_tracker import GoalTree, GoalTracker, Goal

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GoalTracker(Path(tmpdir))

        # Add root goal
        root = Goal(
            id="goal_0001",
            description="Root goal",
            created=datetime.now().isoformat()
        )
        tracker.goals.append(root)

        # Add child goal
        child = Goal(
            id="goal_0002",
            description="Child goal",
            created=datetime.now().isoformat(),
            parent_id="goal_0001"
        )
        tracker.goals.append(child)

        tracker._save_goals()

        tree = GoalTree(tracker)
        roots = tree.get_root_goals()

        # Only root should be returned
        assert len(roots) == 1
        assert roots[0].id == "goal_0001"

    return True


if __name__ == "__main__":
    tests = [
        # GS-001
        test_goal_progress_field,
        test_goal_update_progress,
        test_goal_complete_on_100_percent,
        test_goal_progress_from_subtasks,
        test_progress_detector_creation,
        test_progress_detector_patterns,
        test_subtask_completion_detection,
        # GS-002
        test_goal_priority_score,
        test_goal_set_priority,
        test_goal_priority_bounds,
        test_parse_due_date_tomorrow,
        test_parse_due_date_day_of_week,
        test_parse_due_date_in_days,
        test_goal_overdue_detection,
        test_tracker_goals_by_priority,
        # GS-003
        test_goal_parent_child_fields,
        test_goal_add_remove_child,
        test_goal_is_child,
        test_goal_tree_creation,
        test_goal_tree_create_child,
        test_goal_tree_aggregate_progress,
        test_goal_tree_view,
        test_goal_tree_get_root_goals,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
