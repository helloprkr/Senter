---
name: ralph-wiggums
description: Autonomous iterative development loop. ALWAYS USE when asked to "use Ralph Wiggums", implement requirements autonomously, or build features iteratively. This skill runs ENTIRELY within Claude Code (no API calls) for fully autonomous execution.
---

# Ralph Wiggums

**Fully autonomous development loop powered by Claude Code (subscription-based, NO API).**

**STOP. Before writing ANY code, you MUST complete the Initialization Protocol below.**

---

## Initialization Protocol (REQUIRED)

When this skill is triggered, execute these steps IN ORDER before any implementation:

### Step 1: Create Directory Structure
```bash
mkdir -p scripts/ralph
```

### Step 2: Detect Project Stack
Examine the project to identify:
- Language (check for tsconfig.json, package.json, requirements.txt, go.mod, Cargo.toml)
- Framework (next.config.*, vite.config.*, manage.py, etc.)
- Test runner (vitest.config.*, jest.config.*, pytest.ini, etc.)
- Available npm/pip scripts for typecheck, test, lint

See `references/stack-detection.md` for detailed detection logic.

### Step 3: Generate prd.json
Create `scripts/ralph/prd.json` with this structure:

```json
{
  "projectName": "[NAME FROM REQUIREMENTS]",
  "branchName": "ralph/[feature-name]",
  "stack": {
    "language": "[detected]",
    "framework": "[detected]",
    "testRunner": "[detected]"
  },
  "verification": {
    "typecheck": "[command]",
    "unitTest": "[command]",
    "integrationTest": "[command]",
    "lint": "[command]"
  },
  "userStories": [
    // GENERATE ATOMIC STORIES FROM REQUIREMENTS - see Step 4
  ]
}
```

### Step 4: Decompose Requirements into Atomic Stories
For EACH feature in the requirements, create stories following these rules:

**Story MUST be completable in ONE iteration:**
- Changes ≤5 files
- Has ≤5 acceptance criteria
- NO words like "entire", "complete", "full system"

**Story format:**
```json
{
  "id": "US-001",
  "title": "[Imperative verb] [specific outcome]",
  "acceptanceCriteria": [
    "Specific observable outcome",
    "Another specific outcome",
    "typecheck passes",
    "unit tests pass"
  ],
  "testRequirements": {
    "unit": ["test_function_scenario"],
    "integration": [],
    "visual": false
  },
  "priority": 1,
  "passes": false,
  "notes": ""
}
```

**Decomposition examples:**
- ❌ "Build authentication system" 
- ✅ "Add login form UI" → "Add form validation" → "Add auth API endpoint" → "Add session handling"

See `references/story-decomposition.md` for detailed patterns.

### Step 5: Generate progress.txt
Create `scripts/ralph/progress.txt`:

```markdown
# Ralph Progress Log
Started: [CURRENT DATE]
Project: [PROJECT NAME]

## Codebase Patterns
- Stack: [language]/[framework]
- Test runner: [test runner]
- (Add patterns as discovered)

## Key Files
- (List critical files identified during analysis)

---
```

### Step 6: Create Git Branch
```bash
git checkout -b ralph/[feature-name]
```

### Step 7: Begin Autonomous Execution
**IMMEDIATELY after initialization, begin the autonomous loop below. Do NOT wait for user confirmation.**

---

## 🔄 AUTONOMOUS EXECUTION LOOP

**THIS IS THE CORE OF RALPH WIGGUMS.**

After completing initialization, execute this loop continuously until ALL stories pass:

```
┌─────────────────────────────────────────────────────────────┐
│  RALPH WIGGUMS AUTONOMOUS LOOP                              │
│  Runs entirely within Claude Code session (NO API calls)    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LOOP:                                                      │
│    1. Read scripts/ralph/prd.json                          │
│    2. Read scripts/ralph/progress.txt (check patterns!)    │
│    3. Find highest priority story where passes=false       │
│    4. If NO incomplete stories → COMPLETE, stop            │
│    5. Implement the ONE selected story                     │
│    6. Run ALL verification commands                        │
│    7. If PASS → commit, update prd.json, log to progress   │
│    8. If FAIL → fix (3 attempts), then decompose if stuck  │
│    9. GOTO step 1 (continue immediately, no user input)    │
│                                                             │
│  EXIT CONDITIONS:                                           │
│    - All stories pass → output <promise>COMPLETE</promise>  │
│    - Cannot proceed → output <promise>STUCK</promise>       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Autonomous Loop - Detailed Steps

**For EACH iteration, execute these steps WITHOUT waiting for user input:**

#### 1. Read State
```bash
cat scripts/ralph/prd.json      # Find stories
cat scripts/ralph/progress.txt  # Check patterns FIRST
git status                      # Verify branch
```

#### 2. Select Story
Pick highest priority story where `"passes": false` AND `"blocked"` is not `true`.

**COMPLETION CHECK:**
If ALL stories have `"passes": true`:
```
<promise>COMPLETE</promise>
```
**STOP HERE** - work is done. Do not continue.

#### 3. Implement ONE Story
- Follow patterns from progress.txt "Codebase Patterns" section
- Write tests BEFORE or DURING implementation (TDD)
- Keep changes minimal and focused
- Do NOT implement multiple stories in one iteration

#### 4. Verify (ALL MUST PASS)
Run each verification command from prd.json:
```bash
# Run the exact commands specified in prd.json verification section
# Example for Python:
python -m py_compile main.py  # typecheck
python -m pytest tests/ -v    # unitTest
python -m ruff check .        # lint

# Example for TypeScript/Node:
npm run typecheck
npm test
npm run lint
```

If `visual` is `true` in testRequirements:
- Start dev server if needed
- Navigate to relevant page
- Verify UI matches acceptance criteria

#### 5. On Success: Record and Commit
```bash
git add -A
git commit -m "feat: [US-XXX] - [Title]"
```

Update `scripts/ralph/prd.json`:
- Set `"passes": true` for completed story

Append to `scripts/ralph/progress.txt`:
```markdown
## [DATE] - US-XXX: [Title]
- Implemented: [what was done]
- Files: [list of changed files]
- **Learnings:** [patterns/gotchas discovered]
---
```

If you discovered a reusable pattern, add it to "## Codebase Patterns" at the TOP of progress.txt.

#### 6. On Failure: Recovery Protocol
If verification fails:
1. Attempt fix (up to 3 times per story)
2. If stuck after 3 attempts:
   - Decompose story into 2-3 smaller sub-stories
   - Update prd.json with sub-stories at higher priority
   - Mark original story with `"decomposed": true`
   - Log failure in progress.txt
3. Continue with first sub-story

See `references/failure-recovery.md` for detailed recovery procedures.

#### 7. Continue Loop
**CRITICAL: After completing steps 1-6, IMMEDIATELY return to step 1.**
- Do NOT wait for user input
- Do NOT ask for confirmation
- Continue looping until ALL stories pass or you are STUCK

---

## Stop Conditions

| Signal | Meaning |
|--------|---------|
| `<promise>COMPLETE</promise>` | All stories pass. Work is done. |
| `<promise>STUCK</promise>` | Cannot proceed. Human review needed. |

---

## Pause/Resume (Optional)

If you need to pause mid-execution:
```bash
touch scripts/ralph/.ralph-pause
```

Check for pause at start of each iteration:
```bash
if [ -f scripts/ralph/.ralph-pause ]; then
  echo "PAUSED - remove .ralph-pause to continue"
  # Wait or exit
fi
```

Remove to resume:
```bash
rm scripts/ralph/.ralph-pause
```

---

## Verification Checklist

Before marking ANY story as `"passes": true`, confirm:
- [ ] Typecheck passes
- [ ] Unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Lint passes
- [ ] Visual verification (if `"visual": true`)
- [ ] Changes committed to git

---

## Critical Rules

1. **NEVER skip initialization** — prd.json MUST exist before coding
2. **ONE story per iteration** — never implement multiple stories at once
3. **VERIFY before committing** — never commit failing code
4. **LOG learnings** — progress.txt is memory for future iterations
5. **DECOMPOSE when stuck** — 3 failures = split into smaller stories
6. **CONTINUE AUTONOMOUSLY** — do NOT wait for user input between iterations

---

## Quick Reference

| File | Purpose |
|------|---------|
| `scripts/ralph/prd.json` | Stories, criteria, pass/fail status |
| `scripts/ralph/progress.txt` | Learnings, patterns, session memory |
| `scripts/ralph/.ralph-pause` | Create to pause, delete to resume |

---

## Reference Documents

Load these as needed for detailed guidance:
- `references/stack-detection.md` — How to detect project tech stack
- `references/story-decomposition.md` — How to break down requirements
- `references/test-strategies.md` — Testing patterns per language/framework
- `references/failure-recovery.md` — Recovery protocols for stuck/flaky/blocked stories

---

## Execution Summary

When Ralph Wiggums is invoked:

1. **Initialize** (Steps 1-6) — Create scaffolding, detect stack, generate stories
2. **Loop** (Step 7) — Execute autonomous loop until complete
3. **Signal** — Output `<promise>COMPLETE</promise>` or `<promise>STUCK</promise>`

**All execution happens within your Claude Code session. No external API calls. No ralph.sh script. Pure autonomous iteration.**
