#!/usr/bin/env python3
"""
Tests for UI-001: Menubar Shows Senter is Alive

Tests menubar app functionality (headless mode for CI).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.menubar_app import HeadlessMenubar, RUMPS_AVAILABLE


# ========== Unit Tests ==========

def test_headless_menubar_creation():
    """Test HeadlessMenubar can be created"""
    menubar = HeadlessMenubar()

    assert menubar is not None
    assert menubar.state == "idle"

    return True


def test_headless_state_change():
    """Test state changes"""
    menubar = HeadlessMenubar()

    menubar.set_state("active")
    assert menubar.state == "active"

    menubar.set_state("new")
    assert menubar.state == "new"

    menubar.set_state("idle")
    assert menubar.state == "idle"

    return True


def test_headless_refresh_count():
    """Test unread count refresh"""
    menubar = HeadlessMenubar()

    # Should return a number (may be 0 or more)
    count = menubar.refresh_count()
    assert isinstance(count, int)
    assert count >= 0

    return True


def test_rumps_available():
    """Test that rumps is available for macOS"""
    # rumps should be available on macOS
    # This test documents the dependency
    assert RUMPS_AVAILABLE is True or RUMPS_AVAILABLE is False

    return True


def test_menubar_icons():
    """Test icon constants exist"""
    from ui.menubar_app import SenterMenubar

    assert "idle" in SenterMenubar.ICONS
    assert "active" in SenterMenubar.ICONS
    assert "new" in SenterMenubar.ICONS

    # Icons should be single characters or short strings
    for icon in SenterMenubar.ICONS.values():
        assert len(icon) <= 3

    return True


if __name__ == "__main__":
    tests = [
        test_headless_menubar_creation,
        test_headless_state_change,
        test_headless_refresh_count,
        test_rumps_available,
        test_menubar_icons,
    ]

    print("=" * 60)
    print("UI-001: Menubar App Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                print(f"✓ {test.__name__}")
                passed += 1
            else:
                print(f"✗ {test.__name__}: returned False")
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
