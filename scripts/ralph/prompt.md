# Ralph Wiggums Iteration Protocol

You are executing one iteration of the Ralph Wiggums autonomous development loop.

## Your Task

1. Read the current state from `scripts/ralph/prd.json` and `scripts/ralph/progress.txt`
2. Select the highest priority user story where `"passes": false`
3. Implement ONLY that ONE story
4. Verify all checks pass
5. Commit and update status

## Verification Commands

Run these commands and ALL must pass before marking story complete:

```bash
# Typecheck
python -m py_compile senter.py core/*.py memory/*.py intelligence/*.py coupling/*.py evolution/*.py

# Unit tests
python -m pytest tests/ -v --tb=short

# Integration test
python -c "from core.engine import Senter; print('Import OK')"

# Lint
python -m ruff check . --fix --ignore E501
```

## Iteration Steps

### Step 1: Read State
```bash
cat scripts/ralph/prd.json
cat scripts/ralph/progress.txt
```

### Step 2: Select Story
Pick the highest priority story where `"passes": false`.

If ALL stories have `"passes": true`:
- Output: `<promise>COMPLETE</promise>`
- STOP - work is done

### Step 3: Implement Story
- Follow patterns documented in progress.txt
- Write tests FIRST or during implementation
- Keep changes focused on the single story
- Do NOT implement multiple stories

### Step 4: Verify
```bash
python -m py_compile senter.py core/*.py memory/*.py intelligence/*.py coupling/*.py evolution/*.py
python -m pytest tests/ -v --tb=short
python -c "from core.engine import Senter; print('Import OK')"
python -m ruff check . --fix --ignore E501
```

ALL must pass. If any fails:
1. Fix the issue
2. Re-run verification
3. After 3 failed attempts, decompose story into smaller sub-stories

### Step 5: On Success - Commit and Update
```bash
git add -A
git commit -m "feat: [US-XXX] - [Story Title]"
```

Update `scripts/ralph/prd.json`:
- Set `"passes": true` for completed story

Append to `scripts/ralph/progress.txt`:
```markdown
## [DATE] - US-XXX: [Title]
- Implemented: [what was done]
- Files changed: [list]
- Learnings: [patterns/gotchas discovered]
---
```

### Step 6: Continue or Signal
- More stories? End iteration (loop continues)
- All stories pass? Output `<promise>COMPLETE</promise>`
- Stuck? Output `<promise>STUCK</promise>`

## Critical Rules

1. ONE story per iteration - never multiple
2. VERIFY before committing - never commit failing code
3. LOG learnings - progress.txt is memory for future iterations
4. DECOMPOSE when stuck - 3 failures = split into smaller stories

## Project Context

- **Language**: Python 3.10+
- **Framework**: Custom (Senter 3.0)
- **Test Runner**: pytest with pytest-asyncio
- **Lint**: ruff

## Key Files

- `core/engine.py` - Main Senter orchestration
- `intelligence/activity.py` - Activity monitoring
- `intelligence/goals.py` - Goal detection
- `intelligence/proactive.py` - Proactive suggestions
- `evolution/mutations.py` - Mutation engine
- `memory/living_memory.py` - Memory orchestrator
- `memory/procedural.py` - Procedural memory
- `memory/affective.py` - Affective memory
- `coupling/human_model.py` - Human cognitive state
- `daemon/senter_daemon.py` - Background daemon
- `tools/file_ops.py` - File operations
