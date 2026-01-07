#!/usr/bin/env python3
"""
Test script for Senter TUI components
"""

import sys
import os

# Add paths for imports
script_dir = os.path.dirname(__file__)
senter_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, senter_dir)


def test_imports():
    """Test that all components can be imported"""
    print("Testing imports...")

    try:
        from senter_app import SenterApp

        print("✅ senter_app imported")
    except ImportError as e:
        print(f"❌ senter_app import failed: {e}")

    try:
        from senter_widgets import SidebarItem, EditableListItem

        print("✅ senter_widgets imported")
    except ImportError as e:
        print(f"❌ senter_widgets import failed: {e}")

    try:
        from background_processor import get_background_manager

        print("✅ background_processor imported")
    except ImportError as e:
        print(f"❌ background_processor import failed: {e}")

    try:
        from model_server_manager import ModelServerManager

        print("✅ model_server_manager imported")
    except ImportError as e:
        print(f"❌ model_server_manager import failed: {e}")


def test_basic_functionality():
    """Test basic functionality"""
    print("\\nTesting basic functionality...")

    # Test sidebar item creation
    try:
        from senter_widgets import SidebarItem

        item = SidebarItem("test", "Test Item", "A test item")
        print(f"✅ SidebarItem created: {item.title}")
    except Exception as e:
        print(f"❌ SidebarItem creation failed: {e}")

    # Test background manager
    try:
        from background_processor import get_background_manager

        manager = get_background_manager()
        status = manager.get_status()
        print(f"✅ Background manager status: {status['running']}")
    except Exception as e:
        print(f"❌ Background manager test failed: {e}")


def main():
    """Run all tests"""
    print("🧪 Senter Component Tests")
    print("=" * 40)

    test_imports()
    test_basic_functionality()

    print("\\n🎯 Test complete!")
    print("\\nTo run the full TUI: python senter_app.py")


if __name__ == "__main__":
    main()
