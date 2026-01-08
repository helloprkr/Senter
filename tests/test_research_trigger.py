#!/usr/bin/env python3
"""
Tests for Autonomous Research Trigger - US-005 acceptance criteria:
1. Scheduler job 'background_research' triggers research tasks
2. Research topics extracted from recent user queries
3. Research tasks added to research_tasks queue
4. At least one research task generated per hour when daemon running
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_background_research_job_exists():
    """Test that scheduler has background_research job configured"""
    from scheduler.action_scheduler import ActionScheduler, TriggerType
    from daemon.message_bus import MessageBus
    from multiprocessing import Event

    bus = MessageBus()
    bus.start()

    try:
        shutdown = Event()
        senter_root = Path(__file__).parent.parent

        scheduler = ActionScheduler(
            check_interval=60,
            message_bus=bus,
            shutdown_event=shutdown,
            senter_root=senter_root
        )

        # Create default jobs
        scheduler._create_default_jobs()

        # Check background_research job exists
        assert "background_research" in scheduler.jobs, \
            "Scheduler should have background_research job"

        job = scheduler.jobs["background_research"]
        assert job.job_type == "research", \
            "Job type should be 'research'"
        assert job.trigger_type == TriggerType.INTERVAL, \
            "Job should have interval trigger"
        assert job.trigger_config.get("seconds", 0) == 3600, \
            "Job should trigger every hour (3600 seconds)"

    finally:
        bus.stop()

    print("✓ test_background_research_job_exists PASSED")
    return True


def test_topic_extraction_from_queries():
    """Test that topics can be extracted from user queries"""
    from scheduler.research_trigger import ResearchTopicExtractor
    import tempfile
    import sqlite3

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        db_dir = senter_root / "data" / "learning"
        db_dir.mkdir(parents=True)
        db_path = db_dir / "behavior.db"

        # Create test database with sample queries
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                event_type TEXT,
                timestamp REAL,
                session_id TEXT,
                data TEXT
            );

            CREATE TABLE patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_key TEXT,
                count INTEGER,
                last_seen REAL,
                metadata TEXT,
                UNIQUE(pattern_type, pattern_key)
            );
        """)

        # Insert test queries
        now = time.time()
        test_queries = [
            {"query": "Help me write Python code", "topics": ["coding"]},
            {"query": "Research AI trends", "topics": ["research"]},
            {"query": "Explain machine learning", "topics": ["learning"]},
            {"query": "Debug my function", "topics": ["coding"]},
        ]

        for query in test_queries:
            conn.execute(
                "INSERT INTO events (event_type, timestamp, data) VALUES (?, ?, ?)",
                ("query", now, json.dumps(query))
            )

        # Insert patterns
        conn.execute(
            "INSERT INTO patterns (pattern_type, pattern_key, count, last_seen) VALUES (?, ?, ?, ?)",
            ("topic", "coding", 10, now)
        )
        conn.execute(
            "INSERT INTO patterns (pattern_type, pattern_key, count, last_seen) VALUES (?, ?, ?, ?)",
            ("topic", "research", 5, now)
        )
        conn.commit()
        conn.close()

        # Test extraction
        extractor = ResearchTopicExtractor(senter_root)

        # Get recent topics
        topics = extractor.get_recent_topics(hours=1)
        assert len(topics) > 0, "Should extract topics from queries"
        assert topics[0]["topic"] == "coding", "Most frequent topic should be coding"
        assert topics[0]["query_count"] >= 2, "Should have at least 2 coding queries"

        # Get patterns
        patterns = extractor.get_top_patterns()
        assert len(patterns) > 0, "Should have patterns"
        assert patterns[0]["topic"] == "coding", "Top pattern should be coding"

    print("✓ test_topic_extraction_from_queries PASSED")
    return True


def test_research_task_generation():
    """Test that research tasks are generated from topics"""
    from scheduler.research_trigger import ResearchTaskGenerator

    generator = ResearchTaskGenerator()

    # Test single task generation
    task = generator.generate_task("coding", ["How to write Python?", "Debug my code"])
    assert "id" in task, "Task should have id"
    assert "description" in task, "Task should have description"
    assert task["topic"] == "coding", "Task should have topic"
    assert task["source"] == "background_research", "Task source should be background_research"
    assert "coding" in task["description"].lower() or "programming" in task["description"].lower(), \
        "Description should mention coding"

    # Test batch generation
    topics = [
        {"topic": "research", "query_count": 5, "sample_queries": ["Research AI"]},
        {"topic": "learning", "query_count": 3, "sample_queries": ["How does X work?"]},
    ]
    tasks = generator.generate_tasks_from_topics(topics, max_tasks=2)
    assert len(tasks) == 2, "Should generate 2 tasks"
    assert tasks[0]["topic"] == "research", "First task should be research"

    print("✓ test_research_task_generation PASSED")
    return True


def test_trigger_research_integration():
    """Test the full trigger_research flow"""
    from scheduler.research_trigger import trigger_background_research
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)

        # Without any data, should return empty
        tasks = trigger_background_research(senter_root)
        assert tasks == [], "Should return empty list with no data"

        # Create learning database with sample data
        db_dir = senter_root / "data" / "learning"
        db_dir.mkdir(parents=True)
        db_path = db_dir / "behavior.db"

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                event_type TEXT,
                timestamp REAL,
                session_id TEXT,
                data TEXT
            );

            CREATE TABLE patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_key TEXT,
                count INTEGER,
                last_seen REAL,
                metadata TEXT,
                UNIQUE(pattern_type, pattern_key)
            );
        """)

        # Insert test data
        now = time.time()
        conn.execute(
            "INSERT INTO events (event_type, timestamp, data) VALUES (?, ?, ?)",
            ("query", now, json.dumps({"query": "Help with Python", "topics": ["coding"]}))
        )
        conn.execute(
            "INSERT INTO patterns (pattern_type, pattern_key, count, last_seen) VALUES (?, ?, ?, ?)",
            ("topic", "productivity", 8, now)
        )
        conn.commit()
        conn.close()

        # Should now generate tasks
        tasks = trigger_background_research(senter_root)
        assert len(tasks) > 0, "Should generate tasks with data"

    print("✓ test_trigger_research_integration PASSED")
    return True


def test_ipc_handler_exists():
    """Test that IPC handler for trigger_research exists"""
    from daemon.ipc_server import IPCServer
    from daemon.ipc_client import IPCClient

    # Check server handler exists
    server = IPCServer()
    assert "trigger_research" in server.handlers, \
        "IPC server should have trigger_research handler"

    # Check client method exists
    client = IPCClient()
    assert hasattr(client, "trigger_research"), \
        "IPC client should have trigger_research method"

    print("✓ test_ipc_handler_exists PASSED")
    return True


def test_research_trigger():
    """Combined test for US-005 acceptance criteria"""
    from scheduler.action_scheduler import ActionScheduler, TriggerType
    from scheduler.research_trigger import (
        ResearchTopicExtractor, ResearchTaskGenerator, trigger_background_research
    )
    from daemon.message_bus import MessageBus
    from daemon.ipc_server import IPCServer
    from multiprocessing import Event

    # 1. Scheduler job exists and triggers hourly
    bus = MessageBus()
    bus.start()
    try:
        shutdown = Event()
        scheduler = ActionScheduler(
            check_interval=60,
            message_bus=bus,
            shutdown_event=shutdown,
            senter_root=Path(__file__).parent.parent
        )
        scheduler._create_default_jobs()
        assert "background_research" in scheduler.jobs
        assert scheduler.jobs["background_research"].trigger_config["seconds"] == 3600
    finally:
        bus.stop()

    # 2. Topics extracted from queries
    generator = ResearchTaskGenerator()
    task = generator.generate_task("coding")
    assert "description" in task

    # 3. Research tasks can be generated
    topics = [{"topic": "research", "query_count": 5, "sample_queries": []}]
    tasks = generator.generate_tasks_from_topics(topics, max_tasks=1)
    assert len(tasks) == 1
    assert tasks[0]["topic"] == "research"

    # 4. IPC handler exists for triggering
    server = IPCServer()
    assert "trigger_research" in server.handlers

    print("✓ test_research_trigger PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Research Trigger Tests (US-005)")
    print("="*60 + "\n")

    all_passed = True

    tests = [
        test_background_research_job_exists,
        test_topic_extraction_from_queries,
        test_research_task_generation,
        test_trigger_research_integration,
        test_ipc_handler_exists,
        test_research_trigger,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)

    sys.exit(0 if all_passed else 1)
