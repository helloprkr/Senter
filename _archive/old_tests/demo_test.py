#!/usr/bin/env python3
"""
Senter Demo Verification Script
Tests the full interaction flow with configured LLM backend
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_model_config():
    """Verify model is configured"""
    print("\n=== Checking Model Configuration ===")
    config_path = Path("config/user_profile.json")

    if not config_path.exists():
        print("  [FAIL] No config/user_profile.json found")
        print("         Create one with your model settings")
        return False

    config = json.loads(config_path.read_text())
    model_config = config.get("central_model")

    if not model_config:
        print("  [FAIL] No 'central_model' in config")
        return False

    model_type = model_config.get("type", "unknown")
    print(f"  [OK] Model type: {model_type}")

    if model_type == "openai":
        endpoint = model_config.get("endpoint", "NOT SET")
        model_name = model_config.get("model_name", "NOT SET")
        print(f"  [OK] Endpoint: {endpoint}")
        print(f"  [OK] Model: {model_name}")

        # Check if Ollama endpoint
        if "11434" in endpoint:
            print("  [OK] Detected Ollama backend")
    elif model_type == "gguf":
        path = model_config.get("path", "NOT SET")
        print(f"  [INFO] GGUF path: {path}")
        if not Path(path).exists():
            print(f"  [WARN] GGUF file not found at path")

    return True


def test_ollama_connection():
    """Test that Ollama is running and responsive"""
    print("\n=== Testing Ollama Connection ===")
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"  [OK] Ollama is running with {len(models)} models")
            for m in models[:3]:
                print(f"       - {m.get('name')}")
            return True
        else:
            print(f"  [FAIL] Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Cannot connect to Ollama: {e}")
        return False


def test_omniagent_import():
    """Test that the main agent can be imported"""
    print("\n=== Testing OmniAgent Import ===")
    try:
        from Functions.omniagent import SenterOmniAgent

        print("  [OK] SenterOmniAgent can be imported")
        return True
    except Exception as e:
        print(f"  [FAIL] OmniAgent import failed: {e}")
        return False


def test_focus_loading():
    """Test that focuses can be loaded"""
    print("\n=== Testing Focus Loading ===")
    try:
        from Focuses.senter_md_parser import SenterMdParser

        parser = SenterMdParser(Path("."))
        focuses = parser.list_all_focuses()

        if not focuses:
            print("  [FAIL] No focuses found")
            return False

        print(f"  [OK] Found {len(focuses)} focuses:")
        for f in focuses:
            # Try to load each focus config
            try:
                config = parser.load_focus_config(f)
                status = "[OK]" if config else "[WARN]"
            except Exception:
                status = "[WARN]"
            print(f"       {status} {f}")

        return True
    except Exception as e:
        print(f"  [FAIL] Focus loading failed: {e}")
        return False


def test_router():
    """Test that routing configuration exists"""
    print("\n=== Testing Router ===")
    try:
        router_path = Path("Focuses/internal/Router/SENTER.md")
        if router_path.exists():
            print("  [OK] Router SENTER.md exists")
            content = router_path.read_text()
            if "focus" in content.lower() and "route" in content.lower():
                print("  [OK] Router contains routing instructions")
                return True
        print("  [WARN] Router may not be properly configured")
        return True  # Non-fatal
    except Exception as e:
        print(f"  [WARN] Router check warning: {e}")
        return True


def test_web_search():
    """Test web search import"""
    print("\n=== Testing Web Search ===")
    try:
        from Functions.web_search import search_web

        # Just test import, not actual search (API may return empty)
        print("  [OK] Web search function importable")
        return True
    except Exception as e:
        print(f"  [WARN] Web search warning: {e}")
        return True  # Non-fatal for demo


def test_cli_help():
    """Test CLI can show help"""
    print("\n=== Testing CLI ===")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/senter.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "Senter CLI" in result.stdout:
            print("  [OK] CLI help works")
            return True
        else:
            print(f"  [FAIL] CLI help failed")
            return False
    except Exception as e:
        print(f"  [FAIL] CLI test failed: {e}")
        return False


def test_llm_response():
    """Test that LLM actually responds to a query"""
    print("\n=== Testing LLM Response ===")
    try:
        import requests

        # Direct Ollama test
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "Say 'test passed' in 3 words"}],
                "max_tokens": 20,
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  [OK] LLM responded: {content[:50]}...")
            return True
        else:
            print(f"  [FAIL] LLM returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] LLM test failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  SENTER DEMO VERIFICATION")
    print("=" * 55)

    results = {
        "Model Config": check_model_config(),
        "Ollama Connection": test_ollama_connection(),
        "OmniAgent Import": test_omniagent_import(),
        "Focus Loading": test_focus_loading(),
        "Router": test_router(),
        "Web Search": test_web_search(),
        "CLI Help": test_cli_help(),
        "LLM Response": test_llm_response(),
    }

    print("\n" + "=" * 55)
    print("  DEMO READINESS SUMMARY")
    print("=" * 55)

    # Critical components that must pass
    critical = ["Model Config", "Ollama Connection", "OmniAgent Import", "Focus Loading", "LLM Response"]
    critical_pass = all(results.get(c, False) for c in critical)

    for name, status in results.items():
        symbol = "[PASS]" if status else "[FAIL]"
        critical_marker = "*" if name in critical else " "
        print(f"  {symbol}{critical_marker} {name}")

    print("\n  (* = critical for demo)")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Score: {passed}/{total}")

    if critical_pass:
        print("\n  " + "=" * 50)
        print("  DEMO READY - All critical components working!")
        print("  " + "=" * 50)
        print("\n  To start demo:")
        print("    python3 scripts/senter.py      # CLI mode")
        print("    python3 scripts/senter_app.py  # TUI mode")
        print("\n  Demo commands:")
        print("    /list              # Show focuses")
        print("    /focus coding      # Switch focus")
        print("    What is Python?    # Ask a question")
        print("    /exit              # Clean exit")
    else:
        print("\n  " + "=" * 50)
        print("  NOT DEMO READY - Fix critical issues above")
        print("  " + "=" * 50)

    return 0 if critical_pass else 1


if __name__ == "__main__":
    sys.exit(main())
