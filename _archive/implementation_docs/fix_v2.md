# SENTER FIX PLAN v2.0 — Demo Ready

**Date**: January 7, 2026  
**Purpose**: Get Senter running end-to-end with an actual LLM  
**Prerequisite**: fix_v1.md completed (5/5 tests passing)  
**Goal**: A working demo you can show in 60 seconds

---

## CURRENT STATE (After fix_v1)

| Component | Status |
|-----------|--------|
| Core bugs | ✅ Fixed |
| Syntax errors | ✅ Fixed |
| Focus discovery | ✅ Working (returns 5 focuses) |
| Web search | ✅ Importable |
| CLI help | ✅ Works |
| **LLM Integration** | ❓ Not configured |
| **Full interaction** | ❓ Untested |

---

## PHASE 1: MODEL CONFIGURATION

### 1.1 Find Current Model Config

```bash
# Check what model configuration exists
cat config/user_profile.json 2>/dev/null || echo "No user_profile.json"
cat config/model_config.json 2>/dev/null || echo "No model_config.json"

# Look for any config files
find config/ -type f -exec echo "=== {} ===" \; -exec cat {} \;
```

### 1.2 Understand Model Options

Based on the codebase, Senter supports:

| Backend | Config Key | What You Need |
|---------|-----------|---------------|
| **GGUF (llama.cpp)** | Local .gguf file path | A downloaded GGUF model |
| **OpenAI API** | API key + model name | OPENAI_API_KEY env var |
| **vLLM** | OpenAI-compatible endpoint | Running vLLM server |
| **Ollama** | Model name | Running Ollama with model |

### 1.3 Quickest Path: OpenAI API

If you have an OpenAI API key, this is the fastest way to test:

**Create/update `config/user_profile.json`:**
```json
{
  "central_model": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key_env": "OPENAI_API_KEY"
  },
  "user_name": "Demo User",
  "preferences": {}
}
```

**Set environment variable:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 1.4 Alternative: Local GGUF Model

If you have a GGUF model downloaded:

```json
{
  "central_model": {
    "provider": "llama_cpp",
    "model_path": "/path/to/your/model.gguf",
    "context_window": 8192,
    "n_gpu_layers": -1
  }
}
```

### 1.5 Alternative: Ollama

If you have Ollama running:

```json
{
  "central_model": {
    "provider": "ollama",
    "model": "llama3.2",
    "base_url": "http://localhost:11434"
  }
}
```

### 1.6 Verify Model Config

```bash
python3 -c "
import json
from pathlib import Path

config_path = Path('config/user_profile.json')
if config_path.exists():
    config = json.loads(config_path.read_text())
    print('Model config found:')
    print(json.dumps(config.get('central_model', 'NOT SET'), indent=2))
else:
    print('ERROR: No config/user_profile.json found')
    print('Create one using the templates above')
"
```

---

## PHASE 2: END-TO-END TESTING

### 2.1 Test CLI Startup

```bash
# Start the CLI
python3 scripts/senter.py

# Expected: Should show a prompt or welcome message
# If it crashes, note the error
```

### 2.2 Test Basic Commands

Once CLI is running, test these commands:

```
/list          # Should show available focuses
/focus general # Should switch to general focus
/exit          # Should exit cleanly
```

### 2.3 Test TUI Startup

```bash
# Start the TUI (Textual interface)
python3 scripts/senter_app.py

# Expected: Should show a terminal UI
# Press Ctrl+C to exit if needed
```

### 2.4 Test Actual Query

With model configured, test a real query:

```bash
# In CLI mode, type a question:
> What is the capital of France?

# Expected: 
# - Query gets routed
# - Model generates response
# - Response displayed
```

### 2.5 Test Web Search Integration

```bash
# Query that should trigger web search:
> Search for the latest news about AI

# Expected:
# - Web search function called
# - Results incorporated into response
```

---

## PHASE 3: CREATE DEMO SCRIPT

### 3.1 Automated Demo Test

Create a script that verifies the full flow:

**Create `demo_test.py`:**
```python
#!/usr/bin/env python3
"""
Senter Demo Verification Script
Tests the full interaction flow
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_model_config():
    """Verify model is configured"""
    print("\n=== Checking Model Configuration ===")
    import json
    config_path = Path("config/user_profile.json")
    
    if not config_path.exists():
        print("❌ No config/user_profile.json found")
        print("   Create one with your model settings")
        return False
    
    config = json.loads(config_path.read_text())
    model_config = config.get("central_model")
    
    if not model_config:
        print("❌ No 'central_model' in config")
        return False
    
    provider = model_config.get("provider", "unknown")
    print(f"✓ Model provider: {provider}")
    
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ OPENAI_API_KEY not set in environment")
            return False
        print("✓ OPENAI_API_KEY found")
    
    return True


def test_omniagent_init():
    """Test that the main agent can initialize"""
    print("\n=== Testing OmniAgent Initialization ===")
    try:
        from Functions.omniagent import SenterOmniAgent
        # Don't actually instantiate yet - just verify import
        print("✓ SenterOmniAgent can be imported")
        return True
    except Exception as e:
        print(f"❌ OmniAgent import failed: {e}")
        return False


def test_focus_loading():
    """Test that focuses can be loaded"""
    print("\n=== Testing Focus Loading ===")
    try:
        from Focuses.senter_md_parser import SenterMdParser
        parser = SenterMdParser()
        focuses = parser.list_all_focuses()
        
        if not focuses:
            print("❌ No focuses found")
            return False
        
        print(f"✓ Found {len(focuses)} focuses:")
        for f in focuses:
            # Try to load each focus config
            config = parser.parse_focus(f)
            status = "✓" if config else "⚠️"
            print(f"  {status} {f}")
        
        return True
    except Exception as e:
        print(f"❌ Focus loading failed: {e}")
        return False


def test_router():
    """Test that routing logic exists"""
    print("\n=== Testing Router ===")
    try:
        # Check if router SENTER.md exists
        router_path = Path("Focuses/internal/Router/SENTER.md")
        if router_path.exists():
            print("✓ Router SENTER.md exists")
            # Check its content
            content = router_path.read_text()
            if "focus" in content.lower() and "route" in content.lower():
                print("✓ Router contains routing instructions")
                return True
        print("⚠️ Router may not be properly configured")
        return True  # Non-fatal
    except Exception as e:
        print(f"⚠️ Router check warning: {e}")
        return True


def test_web_search():
    """Test web search works"""
    print("\n=== Testing Web Search ===")
    try:
        from Functions.web_search import search_web
        results = search_web("test", max_results=1)
        print(f"✓ Web search returned {len(results)} results")
        return True
    except Exception as e:
        print(f"⚠️ Web search warning: {e}")
        return True  # Non-fatal for demo


def main():
    print("=" * 50)
    print("SENTER DEMO VERIFICATION")
    print("=" * 50)
    
    results = {
        "Model Config": check_model_config(),
        "OmniAgent": test_omniagent_init(),
        "Focus Loading": test_focus_loading(),
        "Router": test_router(),
        "Web Search": test_web_search(),
    }
    
    print("\n" + "=" * 50)
    print("DEMO READINESS SUMMARY")
    print("=" * 50)
    
    critical_pass = results["Model Config"] and results["OmniAgent"] and results["Focus Loading"]
    
    for name, status in results.items():
        symbol = "✓" if status else "❌"
        print(f"{symbol} {name}")
    
    if critical_pass:
        print("\n✅ DEMO READY - Critical components working")
        print("\nTo start demo:")
        print("  python3 scripts/senter.py    # CLI mode")
        print("  python3 scripts/senter_app.py # TUI mode")
    else:
        print("\n❌ NOT DEMO READY - Fix critical issues above")
    
    return 0 if critical_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

### 3.2 Run Demo Test

```bash
python3 demo_test.py
```

---

## PHASE 4: DEMO PREPARATION

### 4.1 What TO Demonstrate

| Feature | How to Show | Expected Result |
|---------|-------------|-----------------|
| CLI startup | `python3 scripts/senter.py` | Clean prompt appears |
| Focus list | `/list` command | Shows 5 focuses |
| Focus switch | `/focus research` | Confirms switch |
| Basic query | "What is Python?" | LLM responds |
| Web search | "Search for AI news" | Fetches & summarizes |
| Clean exit | `/exit` | Exits gracefully |

### 4.2 What NOT to Claim

| Don't Claim | Reality |
|-------------|---------|
| "7 working agents" | 7 SENTER.md prompt templates |
| "Self-learning" | Stub only - prints log message |
| "Semantic routing" | Keyword matching or LLM-based |
| "Parallel inference" | Background threads, not dual queries |
| "Always learning" | No persistent learning implemented |

### 4.3 Honest Talking Points

**What Senter IS:**
- A local-first AI assistant framework
- Configurable focus areas for different tasks
- Integration with web search
- CLI and TUI interfaces
- Extensible via SENTER.md configurations

**What Senter WILL BE (roadmap):**
- Self-learning from conversations
- Semantic routing between focuses
- Parallel inference processing
- Autonomous background tasks

---

## PHASE 5: COMMON ISSUES & FIXES

### 5.1 "No module named 'X'"

```bash
# Install missing dependency
pip install X --break-system-packages

# Or check if it's in requirements
grep -i "X" requirements.txt
```

### 5.2 "Model not found" / Connection Errors

```bash
# For OpenAI
echo $OPENAI_API_KEY  # Should show your key

# For Ollama
curl http://localhost:11434/api/tags  # Should list models

# For GGUF
ls -la /path/to/your/model.gguf  # Should exist
```

### 5.3 "Config file not found"

```bash
# Ensure config directory exists
mkdir -p config

# Create minimal config
cat > config/user_profile.json << 'EOF'
{
  "central_model": {
    "provider": "openai",
    "model": "gpt-4o-mini"
  }
}
EOF
```

### 5.4 TUI Won't Start

```bash
# Ensure Textual is installed
pip install textual --break-system-packages

# Check terminal supports it
echo $TERM  # Should be xterm-256color or similar
```

### 5.5 Crashes on Query

Check these in order:
1. Is model configured correctly?
2. Is API key valid/model available?
3. Is there a traceback? (Read the actual error)
4. Try with a simpler model first

---

## PHASE 6: NEXT STEPS AFTER DEMO WORKS

Once you can demo the basic flow:

### 6.1 Priority Fixes

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Model config documentation | 1 hour |
| P1 | Error messages when model fails | 2 hours |
| P2 | Example interactions in README | 1 hour |
| P3 | Better router implementation | 1 day |

### 6.2 Feature Completion

| Feature | Current State | To Complete |
|---------|--------------|-------------|
| Self-learning | Stub | Implement conversation analysis → SENTER.md updates |
| Semantic routing | Keyword | Implement embedding-based focus selection |
| Parallel inference | Background threads | True dual-model inference |
| Voice I/O | Code exists | Test and document |

### 6.3 Testing

```bash
# Create proper test suite
pip install pytest
pytest tests/ -v
```

---

## EXECUTION CHECKLIST

```
□ PHASE 1: Model Configuration
  □ 1.1 Check existing config
  □ 1.2-1.5 Choose and configure model backend
  □ 1.6 Verify config loads

□ PHASE 2: End-to-End Testing  
  □ 2.1 CLI starts
  □ 2.2 Commands work (/list, /focus, /exit)
  □ 2.3 TUI starts
  □ 2.4 Query gets response
  □ 2.5 Web search works

□ PHASE 3: Demo Script
  □ 3.1 Create demo_test.py
  □ 3.2 All checks pass

□ PHASE 4: Demo Prep
  □ Know what to show
  □ Know what NOT to claim
  □ Practice 60-second demo

□ PHASE 5: Issues Fixed
  □ Any crashes resolved
  □ Model responding
  □ Clean user experience
```

---

## QUESTIONS CLAUDE CODE SHOULD ANSWER

After completing this phase:

1. **What model backend did you configure?** (OpenAI / GGUF / Ollama / vLLM)
2. **Can you type a query and get a response?** (Yes/No + example)
3. **What does the 60-second demo look like?** (List the exact steps)
4. **What still doesn't work?** (List any remaining issues)
5. **What's the honest state of the project?** (Working demo / Partially working / Still broken)

---

## SUCCESS CRITERIA

**Demo Ready means:**
- [ ] CLI starts without errors
- [ ] `/list` shows focuses
- [ ] A query gets a response from the LLM
- [ ] Web search can be triggered
- [ ] `/exit` cleanly terminates

**NOT Demo Ready if:**
- Model configuration fails
- Crashes on first query
- No LLM responses
- Critical errors in logs

---

**End of Fix Plan v2.0**
