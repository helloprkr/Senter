---
name: ralph-wiggums
description: Autonomous iterative development loop. ALWAYS USE when asked to "use Ralph Wiggums", implement requirements autonomously, or build features iteratively. This skill REQUIRES creating scripts/ralph/ scaffolding with prd.json, prompt.md, progress.txt before implementation begins.
---

# Ralph Wiggums

**STOP. Before writing ANY code, you MUST complete the Initialization Protocol below.**

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

### Step 5: Generate prompt.md
Create `scripts/ralph/prompt.md` with iteration instructions.

Use the template from `assets/templates/prompt.md`, replacing:
- `{{TYPECHECK_COMMAND}}` with detected typecheck command
- `{{UNIT_TEST_COMMAND}}` with detected test command
- `{{INTEGRATION_TEST_COMMAND}}` with detected integration test command
- `{{LINT_COMMAND}}` with detected lint command

### Step 6: Generate progress.txt
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

### Step 7: Copy ralph.sh
Copy the loop script from `scripts/ralph.sh` in this skill to `scripts/ralph/ralph.sh` in the project, then make it executable:
```bash
chmod +x scripts/ralph/ralph.sh
```

### Step 8: Create Git Branch
```bash
git checkout -b ralph/[feature-name]
```

### Step 9: Begin Execution Loop
Either:
- Run `./scripts/ralph/ralph.sh [max_iterations]` to start the autonomous loop, OR
- Execute iterations manually by following the prompt.md instructions

---

## Iteration Protocol

Each iteration (whether run via ralph.sh or manually), follow this EXACT sequence:

### 1. Read State
```bash
cat scripts/ralph/prd.json      # Find stories
cat scripts/ralph/progress.txt  # Check patterns FIRST
git status                      # Verify branch
```

### 2. Select Story
Pick highest priority story where `"passes": false`.

If ALL stories have `"passes": true`:
```
<promise>COMPLETE</promise>
```
**STOP HERE** - work is done.

### 3. Implement ONE Story
- Follow patterns from progress.txt
- Write tests BEFORE or DURING implementation
- Keep changes minimal and focused
- Do NOT implement multiple stories

### 4. Verify (ALL MUST PASS)
Run each verification command from prd.json:
```bash
# Example for TypeScript/Node:
npm run typecheck   # Must pass
npm test            # Must pass  
npm run lint        # Must pass
```

### 5. On Success: Record and Commit
```bash
git add -A
git commit -m "feat: [US-XXX] - [Title]"
```

Update `scripts/ralph/prd.json`:
```json
{ "id": "US-XXX", "passes": true, ... }
```

Append to `scripts/ralph/progress.txt`:
```markdown
## [DATE] - US-XXX
- Implemented: [what]
- Files: [list]
- **Learnings:** [patterns/gotchas discovered]
---
```

If you discovered a reusable pattern, add it to "## Codebase Patterns" at the TOP of progress.txt.

### 6. On Failure: Recovery Protocol
If verification fails:
1. Attempt fix (up to 3 times)
2. If stuck after 3 attempts → decompose story into 2-3 smaller sub-stories
3. Update prd.json with sub-stories
4. Log failure in progress.txt
5. Continue with first sub-story

See `references/failure-recovery.md` for detailed recovery procedures.

### 7. Continue or Signal
- More stories remaining? → End iteration (loop continues)
- All stories pass? → Output `<promise>COMPLETE</promise>`
- Stuck, cannot proceed? → Output `<promise>STUCK</promise>`

---

## User Intervention

| Action | Method |
|--------|--------|
| Pause loop | `touch scripts/ralph/.ralph-pause` |
| Resume loop | `rm scripts/ralph/.ralph-pause` |
| Abort | Ctrl+C or kill process |

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

---

## Quick Reference

| File | Purpose |
|------|---------|
| `scripts/ralph/prd.json` | Stories, criteria, pass/fail status |
| `scripts/ralph/prompt.md` | Instructions for each iteration |
| `scripts/ralph/progress.txt` | Learnings, patterns, session memory |
| `scripts/ralph/ralph.sh` | The autonomous loop script |
| `scripts/ralph/.ralph-pause` | Create to pause, delete to resume |

---

## Reference Documents

Load these as needed for detailed guidance:
- `references/stack-detection.md` — How to detect project tech stack
- `references/story-decomposition.md` — How to break down requirements
- `references/test-strategies.md` — Testing patterns per language/framework
- `references/failure-recovery.md` — Recovery protocols for stuck/flaky/blocked stories
