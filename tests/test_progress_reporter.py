#!/usr/bin/env python3
"""
Tests for Progress Reporter (PR-001, PR-002, PR-003)
Tests activity log persistence, digest scheduling, and multi-channel notifications.
"""

import sys
import time
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== PR-001: Activity Log Persistence Tests ==========

def test_activity_log_creation():
    """Test ActivityLog can be created (PR-001)"""
    from reporter.progress_reporter import ActivityLog

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))
        assert log.log_dir.exists()

    return True


def test_activity_log_retention_default():
    """Test ActivityLog has default retention of 30 days (PR-001)"""
    from reporter.progress_reporter import ActivityLog, DEFAULT_RETENTION_DAYS

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))
        assert log.retention_days == DEFAULT_RETENTION_DAYS
        assert log.retention_days == 30

    return True


def test_activity_log_retention_configurable():
    """Test ActivityLog retention is configurable (PR-001)"""
    from reporter.progress_reporter import ActivityLog

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir), retention_days=7)
        assert log.retention_days == 7

    return True


def test_activity_log_persists():
    """Test activity entries persist to disk (PR-001)"""
    from reporter.progress_reporter import ActivityLog, ActivityEntry

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))

        # Log an entry
        entry = ActivityEntry(
            activity_type="test_activity",
            timestamp=time.time(),
            details={"test": "data"},
            source="test"
        )
        log.log(entry)

        # Verify file exists
        log_files = list(Path(tmpdir).glob("activity_*.json"))
        assert len(log_files) == 1

        # Verify content
        content = json.loads(log_files[0].read_text())
        assert len(content) == 1
        assert content[0]["activity_type"] == "test_activity"

    return True


def test_activity_log_cleanup():
    """Test cleanup_old_logs removes old files (PR-001)"""
    from reporter.progress_reporter import ActivityLog

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir), retention_days=5)

        # Create old log files
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        old_file = Path(tmpdir) / f"activity_{old_date}.json"
        old_file.write_text("[]")

        recent_date = datetime.now().strftime("%Y-%m-%d")
        recent_file = Path(tmpdir) / f"activity_{recent_date}.json"
        recent_file.write_text("[]")

        # Run cleanup
        removed = log.cleanup_old_logs()

        assert removed == 1
        assert not old_file.exists()
        assert recent_file.exists()

    return True


def test_activity_log_get_count():
    """Test get_log_count returns correct count (PR-001)"""
    from reporter.progress_reporter import ActivityLog

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))

        # Create test files
        for i in range(3):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            (Path(tmpdir) / f"activity_{date}.json").write_text("[]")

        assert log.get_log_count() == 3

    return True


# ========== PR-002: Configurable Digest Schedule Tests ==========

def test_digest_generator_daily():
    """Test DigestGenerator generates daily digest (PR-002)"""
    from reporter.progress_reporter import DigestGenerator, ActivityLog

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))
        gen = DigestGenerator(log)

        digest = gen.generate_daily_digest()

        assert "Daily Digest" in digest
        assert isinstance(digest, str)

    return True


def test_digest_generator_weekly():
    """Test DigestGenerator generates weekly digest (PR-002)"""
    from reporter.progress_reporter import DigestGenerator, ActivityLog

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))
        gen = DigestGenerator(log)

        digest = gen.generate_weekly_digest()

        assert "Weekly Digest" in digest
        assert isinstance(digest, str)

    return True


def test_digest_generator_period():
    """Test DigestGenerator.generate_digest with period param (PR-002)"""
    from reporter.progress_reporter import DigestGenerator, ActivityLog

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))
        gen = DigestGenerator(log)

        daily = gen.generate_digest(period="daily")
        weekly = gen.generate_digest(period="weekly")

        assert "Daily" in daily
        assert "Weekly" in weekly

    return True


def test_digest_includes_activity_breakdown():
    """Test digest includes activity breakdown (PR-002)"""
    from reporter.progress_reporter import DigestGenerator, ActivityLog, ActivityEntry

    with tempfile.TemporaryDirectory() as tmpdir:
        log = ActivityLog(Path(tmpdir))

        # Add some activities
        for i in range(5):
            log.log(ActivityEntry(
                activity_type="task_completed",
                timestamp=time.time() - 1000,  # Recent
                details={"task": f"Task {i}"},
                source="test"
            ))

        gen = DigestGenerator(log)
        digest = gen.generate_daily_digest()

        # Should mention task_completed
        assert "task_completed" in digest or "activities" in digest.lower()

    return True


# ========== PR-003: Multi-Channel Notification Tests ==========

def test_notification_channel_enum():
    """Test NotificationChannel enum (PR-003)"""
    from reporter.progress_reporter import NotificationChannel

    assert NotificationChannel.DESKTOP.value == "desktop"
    assert NotificationChannel.EMAIL.value == "email"
    assert NotificationChannel.WEBHOOK.value == "webhook"
    assert NotificationChannel.CONSOLE.value == "console"

    return True


def test_notification_config_defaults():
    """Test NotificationConfig has correct defaults (PR-003)"""
    from reporter.progress_reporter import NotificationConfig

    config = NotificationConfig()

    assert config.desktop_enabled is True
    assert config.email_enabled is False
    assert config.webhook_enabled is False
    assert config.email_to is None
    assert config.webhook_url is None

    return True


def test_notification_config_channel_mapping():
    """Test NotificationConfig has channel mapping (PR-003)"""
    from reporter.progress_reporter import NotificationConfig

    config = NotificationConfig()

    assert "task_completed" in config.channel_mapping
    assert "goal_completed" in config.channel_mapping
    assert "digest_ready" in config.channel_mapping

    # goal_completed should notify webhook
    assert "webhook" in config.channel_mapping["goal_completed"]

    return True


def test_notifier_creation():
    """Test Notifier can be created (PR-003)"""
    from reporter.progress_reporter import Notifier, NotificationConfig

    # Default config
    notifier = Notifier()
    assert notifier.config is not None

    # Custom config
    config = NotificationConfig(desktop_enabled=False)
    notifier = Notifier(config)
    assert notifier.config.desktop_enabled is False

    return True


def test_notifier_get_enabled_channels():
    """Test Notifier.get_enabled_channels (PR-003)"""
    from reporter.progress_reporter import Notifier, NotificationConfig

    # Default - only desktop enabled
    notifier = Notifier()
    channels = notifier.get_enabled_channels()
    assert "desktop" in channels
    assert "email" not in channels
    assert "webhook" not in channels

    # All enabled
    config = NotificationConfig(
        desktop_enabled=True,
        email_enabled=True,
        webhook_enabled=True
    )
    notifier = Notifier(config)
    channels = notifier.get_enabled_channels()
    assert len(channels) == 3

    return True


def test_notifier_console_channel():
    """Test Notifier console notification (PR-003)"""
    from reporter.progress_reporter import Notifier, NotificationConfig

    config = NotificationConfig(desktop_enabled=False)
    notifier = Notifier(config)

    # This should not raise
    notifier.notify(
        "Test Title",
        "Test Message",
        event_type="test",
        channels=["console"]
    )

    return True


def test_notifier_respects_event_type():
    """Test Notifier uses channel mapping for event type (PR-003)"""
    from reporter.progress_reporter import Notifier, NotificationConfig

    config = NotificationConfig(
        desktop_enabled=True,
        channel_mapping={
            "test_event": ["console"],
            "default": ["desktop"]
        }
    )
    notifier = Notifier(config)

    # Should use console for test_event
    with patch.object(notifier, '_notify_console') as mock_console:
        notifier.notify("Title", "Message", event_type="test_event")
        mock_console.assert_called_once()

    return True


def test_notifier_webhook_payload_structure():
    """Test webhook notification payload structure (PR-003)"""
    from reporter.progress_reporter import Notifier, NotificationConfig

    config = NotificationConfig(
        webhook_enabled=True,
        webhook_url="https://example.com/hook"
    )
    notifier = Notifier(config)

    # Mock urlopen to capture the request
    captured_payload = None

    def mock_urlopen(request, timeout=None):
        nonlocal captured_payload
        captured_payload = json.loads(request.data.decode('utf-8'))

        class MockResponse:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass

        return MockResponse()

    with patch('urllib.request.urlopen', mock_urlopen):
        notifier._notify_webhook(
            "Test Title",
            "Test Message",
            "test_event",
            {"extra": "data"}
        )

    assert captured_payload is not None
    assert captured_payload["title"] == "Test Title"
    assert captured_payload["message"] == "Test Message"
    assert captured_payload["event_type"] == "test_event"
    assert "timestamp" in captured_payload
    assert captured_payload["data"]["extra"] == "data"

    return True


if __name__ == "__main__":
    tests = [
        # PR-001
        test_activity_log_creation,
        test_activity_log_retention_default,
        test_activity_log_retention_configurable,
        test_activity_log_persists,
        test_activity_log_cleanup,
        test_activity_log_get_count,
        # PR-002
        test_digest_generator_daily,
        test_digest_generator_weekly,
        test_digest_generator_period,
        test_digest_includes_activity_breakdown,
        # PR-003
        test_notification_channel_enum,
        test_notification_config_defaults,
        test_notification_config_channel_mapping,
        test_notifier_creation,
        test_notifier_get_enabled_channels,
        test_notifier_console_channel,
        test_notifier_respects_event_type,
        test_notifier_webhook_payload_structure,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
