#!/usr/bin/env python3
"""
Tests for launchd Service (DI-001)
Tests the launchd plist template and installation scripts.
"""

import sys
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_launchd_plist_exists():
    """Test that plist template exists"""
    senter_root = Path(__file__).parent.parent
    plist_path = senter_root / "scripts" / "com.senter.daemon.plist"

    assert plist_path.exists()

    return True


def test_launchd_plist_valid_xml():
    """Test that plist is valid XML"""
    import xml.etree.ElementTree as ET

    senter_root = Path(__file__).parent.parent
    plist_path = senter_root / "scripts" / "com.senter.daemon.plist"

    # Should parse without error
    tree = ET.parse(plist_path)
    root = tree.getroot()

    assert root.tag == "plist"

    return True


def test_launchd_plist_generation():
    """Test plist placeholder replacement"""
    senter_root = Path(__file__).parent.parent
    plist_path = senter_root / "scripts" / "com.senter.daemon.plist"

    content = plist_path.read_text()

    # Should contain placeholders
    assert "__PYTHON_PATH__" in content
    assert "__SENTER_ROOT__" in content

    # Simulate replacement
    replaced = content.replace("__PYTHON_PATH__", "/usr/bin/python3")
    replaced = replaced.replace("__SENTER_ROOT__", "/opt/senter")

    # Should no longer contain placeholders
    assert "__PYTHON_PATH__" not in replaced
    assert "__SENTER_ROOT__" not in replaced

    # Should contain actual paths
    assert "/usr/bin/python3" in replaced
    assert "/opt/senter" in replaced

    return True


def test_launchd_plist_contains_required_keys():
    """Test that plist contains all required keys"""
    senter_root = Path(__file__).parent.parent
    plist_path = senter_root / "scripts" / "com.senter.daemon.plist"

    content = plist_path.read_text()

    # Required launchd keys
    required_keys = [
        "<key>Label</key>",
        "<key>ProgramArguments</key>",
        "<key>RunAtLoad</key>",
        "<key>KeepAlive</key>",
        "<key>WorkingDirectory</key>",
    ]

    for key in required_keys:
        assert key in content, f"Missing key: {key}"

    return True


def test_launchd_plist_correct_label():
    """Test that plist has correct service label"""
    senter_root = Path(__file__).parent.parent
    plist_path = senter_root / "scripts" / "com.senter.daemon.plist"

    content = plist_path.read_text()

    assert "<string>com.senter.daemon</string>" in content

    return True


def test_install_script_exists():
    """Test that install script exists and is executable"""
    senter_root = Path(__file__).parent.parent
    script_path = senter_root / "scripts" / "install-service.sh"

    assert script_path.exists()

    # Check if executable (on Unix)
    import os
    import stat
    mode = os.stat(script_path).st_mode
    assert mode & stat.S_IXUSR, "Script not executable by owner"

    return True


def test_uninstall_script_exists():
    """Test that uninstall script exists and is executable"""
    senter_root = Path(__file__).parent.parent
    script_path = senter_root / "scripts" / "uninstall-service.sh"

    assert script_path.exists()

    # Check if executable
    import os
    import stat
    mode = os.stat(script_path).st_mode
    assert mode & stat.S_IXUSR, "Script not executable by owner"

    return True


def test_install_script_has_correct_paths():
    """Test that install script references correct files"""
    senter_root = Path(__file__).parent.parent
    script_path = senter_root / "scripts" / "install-service.sh"

    content = script_path.read_text()

    # Should reference the plist template
    assert "com.senter.daemon.plist" in content

    # Should reference LaunchAgents directory
    assert "Library/LaunchAgents" in content

    # Should use launchctl commands
    assert "launchctl load" in content

    return True


def test_uninstall_script_has_correct_logic():
    """Test that uninstall script has proper cleanup logic"""
    senter_root = Path(__file__).parent.parent
    script_path = senter_root / "scripts" / "uninstall-service.sh"

    content = script_path.read_text()

    # Should stop service
    assert "launchctl stop" in content

    # Should unload service
    assert "launchctl unload" in content

    # Should remove plist file
    assert "rm " in content

    return True


if __name__ == "__main__":
    tests = [
        test_launchd_plist_exists,
        test_launchd_plist_valid_xml,
        test_launchd_plist_generation,
        test_launchd_plist_contains_required_keys,
        test_launchd_plist_correct_label,
        test_install_script_exists,
        test_uninstall_script_exists,
        test_install_script_has_correct_paths,
        test_uninstall_script_has_correct_logic,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
