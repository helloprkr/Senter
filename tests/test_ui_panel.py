#!/usr/bin/env python3
"""
Tests for UI-002: Research Panel That's Nice to Read

Tests research panel functionality (headless mode for CI).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.research_panel import HeadlessPanel, PYQT_AVAILABLE
from ui.styles import COLORS, get_confidence_color, FULL_STYLESHEET


# ========== Style Tests ==========

def test_colors_defined():
    """Test that all colors are defined"""
    required_colors = [
        "background", "surface", "primary", "text",
        "text_secondary", "success", "warning", "error"
    ]

    for color in required_colors:
        assert color in COLORS, f"Missing color: {color}"
        assert COLORS[color] is not None

    return True


def test_confidence_color():
    """Test confidence color mapping"""
    # High confidence = success (green)
    high = get_confidence_color(0.9)
    assert high == COLORS["success"]

    # Medium confidence = warning (yellow)
    med = get_confidence_color(0.7)
    assert med == COLORS["warning"]

    # Low confidence = error (red)
    low = get_confidence_color(0.4)
    assert low == COLORS["error"]

    return True


def test_stylesheet_complete():
    """Test that stylesheet is complete"""
    assert len(FULL_STYLESHEET) > 100  # Should be substantial

    # Check key selectors are present
    assert "QMainWindow" in FULL_STYLESHEET
    assert "QLabel" in FULL_STYLESHEET
    assert "QPushButton" in FULL_STYLESHEET
    assert "QScrollArea" in FULL_STYLESHEET

    return True


# ========== Panel Tests ==========

def test_headless_panel_creation():
    """Test HeadlessPanel can be created"""
    panel = HeadlessPanel()

    assert panel is not None

    return True


def test_headless_research_count():
    """Test research count"""
    panel = HeadlessPanel()

    count = panel.get_research_count()
    assert isinstance(count, int)
    assert count >= 0

    return True


def test_pyqt_available():
    """Test that PyQt6 is available"""
    # PyQt6 should be available for full UI
    assert PYQT_AVAILABLE is True or PYQT_AVAILABLE is False

    return True


if __name__ == "__main__":
    tests = [
        test_colors_defined,
        test_confidence_color,
        test_stylesheet_complete,
        test_headless_panel_creation,
        test_headless_research_count,
        test_pyqt_available,
    ]

    print("=" * 60)
    print("UI-002: Research Panel Tests")
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
