# SENTER FIX PLAN v1.0

**Date**: January 7, 2026  
**Purpose**: Get Senter to a demo-able, functional state  
**Context**: This document is for Claude Code to systematically fix the Senter project  
**Estimated Time**: 4-8 hours of focused work

---

## CURRENT STATE SUMMARY

| Category | Status |
|----------|--------|
| **Core Infrastructure** | 60% functional |
| **Critical Bugs** | 3 syntax errors + 1 logic bug that breaks core functionality |
| **"Agents"** | 7 SENTER.md prompt templates (not Python code) |
| **Working Components** | TUI/CLI, Web Search, SENTER.md Parser, OmniAgent |
| **Not Working** | Focus discovery, self-learning (stub), semantic routing |

---

## PHASE 1: CRITICAL BUG FIXES (Do These First)

### 1.1 Fix `list_all_focuses()` - BROKEN CORE FUNCTION

**File**: `Focuses/senter_md_parser.py`  
**Problem**: Function returns empty list immediately, making all code below dead code

**Current broken code (around line 268-277):**
```python
def list_all_focuses(self) -> List[str]:
    """List all available Focus directories"""
    return []  # <-- BUG: Immediately returns empty, everything below is dead code

    focuses = []
    for item in self.focuses_dir.iterdir():
        if item.is_dir() and (item / "SENTER.md").exists():
            focuses.append(item.name)

    return focuses
```

**FIX - Replace with:**
```python
def list_all_focuses(self) -> List[str]:
    """List all available Focus directories"""
    focuses = []
    for item in self.focuses_dir.iterdir():
        if item.is_dir() and (item / "SENTER.md").exists():
            focuses.append(item.name)
    return focuses
```

**Action for Claude Code:**
```bash
# Navigate to the file and remove the premature `return []` line
# The fix is simply deleting line: return []
```

---

### 1.2 Fix Syntax Error in `review_chain.py`

**File**: `Focuses/review_chain.py`  
**Problem**: f-string error - single '}' is not allowed (line 323)

**Likely Issue**: Unescaped curly brace in an f-string

**Action for Claude Code:**
1. Open `Focuses/review_chain.py`
2. Go to line 323 (or search for problematic f-string)
3. Find any `}` that should be `}}` (escaped)
4. Common pattern: JSON in f-strings needs double braces

**Example fix:**
```python
# BROKEN:
f"Expected format: {some_json: value}"

# FIXED:
f"Expected format: {{some_json: value}}"
```

---

### 1.3 Fix Syntax Error in `agent_registry.py`

**File**: `scripts/agent_registry.py`  
**Problem**: unexpected indent (line 1)

**Likely Issue**: 
- File starts with indented code (no class/function definition first)
- OR invisible characters/BOM at file start

**Action for Claude Code:**
1. Open `scripts/agent_registry.py`
2. Check line 1 - should not be indented
3. If there's invisible whitespace, delete and retype
4. Ensure file starts with proper Python (import, class def, or comment)

---

### 1.4 Fix Syntax Error in `function_agent_generator.py`

**File**: `scripts/function_agent_generator.py`  
**Problem**: invalid syntax (line 295)

**Action for Claude Code:**
1. Open `scripts/function_agent_generator.py`
2. Navigate to line 295
3. Common issues:
   - Missing colon after function/class/if/for
   - Unmatched parentheses/brackets
   - Invalid variable names
   - Python 2 syntax (print without parentheses)
4. Fix the syntax error

---

## PHASE 2: VERIFY CORE COMPONENTS WORK

After fixing bugs, run these verification steps:

### 2.1 Syntax Verification

```bash
# Run from project root
python3 -m py_compile Focuses/senter_md_parser.py
python3 -m py_compile Focuses/review_chain.py
python3 -m py_compile scripts/agent_registry.py
python3 -m py_compile scripts/function_agent_generator.py

# Should produce NO output if successful
```

### 2.2 Import Verification

```bash
# Test critical imports
python3 -c "from Focuses.senter_md_parser import SenterMdParser; print('Parser OK')"
python3 -c "from Functions.web_search import search_web; print('Web Search OK')"
python3 -c "from Functions.omniagent import SenterOmniAgent; print('OmniAgent OK')"
python3 -c "from scripts.background_processor import BackgroundTaskManager; print('Background OK')"
```

### 2.3 Focus Discovery Verification

```bash
# After fixing list_all_focuses(), this should return actual focuses
python3 -c "
from Focuses.senter_md_parser import SenterMdParser
parser = SenterMdParser()
focuses = parser.list_all_focuses()
print(f'Found {len(focuses)} focuses: {focuses}')
"
# Expected: Should list coding, creative, general, internal, research, user_personal
```

---

## PHASE 3: MINIMAL DEMO SETUP

### 3.1 Identify Required Model Configuration

The system needs an LLM to function. Check what's configured:

```bash
# Look for model configuration
find . -name "*.json" -o -name "*.yaml" -o -name "*.yml" | xargs grep -l "model" 2>/dev/null
cat config/*.json 2>/dev/null
```

**Claude Code Action**: Document what model the system expects and whether it's configured.

### 3.2 Create Minimal Test Script

Create a test script to verify end-to-end functionality:

**Create file `test_senter_minimal.py`:**
```python
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
        parser = SenterMdParser()
        focuses = parser.list_all_focuses()
        print(f"✓ Parser works - found {len(focuses)} focuses")
        for f in focuses:
            print(f"  - {f}")
        return True
    except Exception as e:
        print(f"✗ Parser failed: {e}")
        return False

def test_web_search():
    """Test web search"""
    print("\n=== Testing Web Search ===")
    try:
        from Functions.web_search import search_web
        results = search_web("test query", max_results=2)
        print(f"✓ Web search works - got {len(results)} results")
        return True
    except Exception as e:
        print(f"✗ Web search failed: {e}")
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
            timeout=10
        )
        if result.returncode == 0:
            print("✓ CLI help works")
            return True
        else:
            print(f"✗ CLI failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    print("=" * 50)
    print("SENTER MINIMAL FUNCTIONALITY TEST")
    print("=" * 50)
    
    results = {
        "Parser": test_parser(),
        "Web Search": test_web_search(),
        "CLI": test_cli()
    }
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Senter core is functional.")
    else:
        print("\n⚠️ Some tests failed. Review issues above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
```

### 3.3 Run Minimal Test

```bash
python3 test_senter_minimal.py
```

---

## PHASE 4: STUB COMPLETION (Optional - For Better Demo)

These are not broken, but are stubs that should be acknowledged or completed:

### 4.1 `_evolve_agents()` Stub

**File**: `scripts/background_processor.py` (line ~255)  
**Current**: Just prints a log message  
**Status**: Acknowledge this is aspirational, not functional

**For demo purposes**: Either:
- Comment it out to avoid confusion
- Add a TODO comment explaining it's planned
- Actually implement basic learning (significant work)

### 4.2 Embedding-based Selection

**File**: `Functions/intelligent_selection.py`  
**Status**: Code exists but may not have embeddings generated

**To verify:**
```bash
# Check if embeddings exist
find . -name "*.npy" -o -name "*embedding*" -o -name "*vector*" | head -20
```

---

## PHASE 5: DEPENDENCY CLEANUP (Optional)

### 5.1 Identify Actually Used Dependencies

```bash
# Find what's actually imported
grep -rh "^import \|^from " --include="*.py" . | sort | uniq -c | sort -rn | head -30
```

### 5.2 Create Minimal requirements.txt

Based on what's actually imported, the minimal requirements are:

```
# Core (definitely needed)
requests>=2.28.0      # For web_search
numpy>=1.24.0         # For embeddings
torch>=2.0.0          # For ML operations
textual>=0.40.0       # For TUI

# Optional (based on model choice)
llama-cpp-python>=0.3.0    # If using GGUF models
openai>=1.0.0              # If using OpenAI API
transformers>=4.40.0       # If using HuggingFace models
```

---

## PHASE 6: DOCUMENTATION HONESTY

### 6.1 Update README Claims

The README should be updated to reflect reality:

**Change from:**
> "✅ Core System: Fully functional"
> "✅ Internal Agents: 7 agents working"

**To:**
> "✅ Core System: CLI and TUI functional with LLM backend"
> "✅ Agent Prompts: 7 SENTER.md prompt templates configured"
> "⚠️ Routing: Uses prompt-based routing (not semantic)"

### 6.2 Clarify Agent Reality

Create or update a notice that "agents" are prompt templates:

```markdown
## How Agents Work

Senter's "agents" are SENTER.md configuration files containing:
- System prompts that instruct the LLM
- YAML configuration for behavior
- Tool definitions

The intelligence comes from the LLM you provide - agents are prompt 
templates that shape LLM behavior, not standalone Python classes.
```

---

## PHASE 7: DEMO PREPARATION CHECKLIST

Before demonstrating Senter, verify:

- [ ] All 4 critical bugs fixed (1.1 - 1.4)
- [ ] Syntax verification passes (2.1)
- [ ] All imports work (2.2)
- [ ] `list_all_focuses()` returns actual focuses (2.3)
- [ ] Minimal test script passes (3.3)
- [ ] LLM is configured and accessible
- [ ] Can run `python scripts/senter.py` without errors
- [ ] Can execute at least one command (`/list`, `/focus`, `/exit`)

---

## EXECUTION ORDER FOR CLAUDE CODE

```
1. FIX CRITICAL BUGS (must do)
   ├── 1.1 Remove `return []` from list_all_focuses()
   ├── 1.2 Fix f-string in review_chain.py line 323
   ├── 1.3 Fix indent in agent_registry.py line 1
   └── 1.4 Fix syntax in function_agent_generator.py line 295

2. VERIFY FIXES (must do)
   ├── 2.1 Run py_compile on all fixed files
   ├── 2.2 Test imports
   └── 2.3 Verify list_all_focuses() returns data

3. CREATE & RUN TEST (recommended)
   ├── 3.2 Create test_senter_minimal.py
   └── 3.3 Run and verify passing

4. DOCUMENT FINDINGS (recommended)
   └── Report what model config exists and what's needed

5. OPTIONAL CLEANUP
   ├── Phase 4: Acknowledge or remove stubs
   ├── Phase 5: Clean dependencies
   └── Phase 6: Update documentation
```

---

## EXPECTED OUTCOME

After completing this fix plan:

| Component | Before | After |
|-----------|--------|-------|
| Focus Discovery | Broken (returns []) | Working |
| Syntax Errors | 3 files broken | 0 errors |
| CLI | May crash | Runs cleanly |
| Demo Ready | No | Yes (with LLM) |

**What This Does NOT Fix:**
- Self-learning (requires actual implementation)
- Semantic routing (requires embedding generation)
- Model availability (user must provide)
- Test coverage (only adds minimal test)

---

## QUESTIONS CLAUDE CODE SHOULD ANSWER

After executing fixes, provide answers to:

1. What model configuration exists and what's needed to run?
2. What's the actual command to start Senter for demo?
3. Are there any additional import errors discovered?
4. What does `list_all_focuses()` return after the fix?
5. Can you successfully execute `/list` in the CLI?

---

**End of Fix Plan v1.0**
