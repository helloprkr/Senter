#!/usr/bin/env python3
"""Minimal Senter functionality test"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_parser():
    """Test SENTER.md parser"""
    print("\n=== Testing Parser ===")
    try:
        from Focuses.senter_md_parser import SenterMdParser

        parser = SenterMdParser(Path("."))
        focuses = parser.list_all_focuses()
        print(f"  Parser works - found {len(focuses)} focuses")
        for f in focuses:
            print(f"    - {f}")
        return True
    except Exception as e:
        print(f"  Parser failed: {e}")
        return False


def test_web_search():
    """Test web search"""
    print("\n=== Testing Web Search ===")
    try:
        from Functions.web_search import search_web

        results = search_web("test query", max_results=2)
        print(f"  Web search works - got {len(results)} results")
        return True
    except Exception as e:
        print(f"  Web search failed: {e}")
        return False


def test_cli():
    """Test CLI can start"""
    print("\n=== Testing CLI ===")
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/senter.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("  CLI help works")
            return True
        else:
            print(f"  CLI failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  CLI test failed: {e}")
        return False


def test_syntax():
    """Test all previously broken files compile"""
    print("\n=== Testing Syntax (Previously Broken Files) ===")
    files_to_check = [
        "Focuses/senter_md_parser.py",
        "Focuses/review_chain.py",
        "scripts/agent_registry.py",
        "scripts/function_agent_generator.py",
    ]

    all_ok = True
    for filepath in files_to_check:
        try:
            with open(filepath, "r") as f:
                compile(f.read(), filepath, "exec")
            print(f"    {filepath}: OK")
        except SyntaxError as e:
            print(f"    {filepath}: SYNTAX ERROR - {e}")
            all_ok = False
        except FileNotFoundError:
            print(f"    {filepath}: FILE NOT FOUND")
            all_ok = False

    return all_ok


def test_focus_config():
    """Test loading a focus configuration"""
    print("\n=== Testing Focus Config Loading ===")
    try:
        from Focuses.senter_md_parser import SenterMdParser

        parser = SenterMdParser(Path("."))
        config = parser.load_focus_config("general")
        if config:
            print(f"  Loaded 'general' focus config")
            print(f"    Keys: {list(config.keys())}")
            return True
        else:
            print("  Could not load 'general' focus config")
            return False
    except Exception as e:
        print(f"  Focus config test failed: {e}")
        return False


def main():
    print("=" * 50)
    print("SENTER MINIMAL FUNCTIONALITY TEST")
    print("=" * 50)

    results = {
        "Syntax Check": test_syntax(),
        "Parser": test_parser(),
        "Focus Config": test_focus_config(),
        "Web Search": test_web_search(),
        "CLI": test_cli(),
    }

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        symbol = "[PASS]" if status else "[FAIL]"
        print(f"{symbol} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n All tests passed! Senter core is functional.")
    else:
        print("\n Some tests failed. Review issues above.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
