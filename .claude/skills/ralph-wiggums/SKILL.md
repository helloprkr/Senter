---
name: ralph-wiggums
description: Autonomous iterative development loop that runs entirely within Claude Code. Decomposes requirements into atomic user stories, implements them one-by-one with rigorous verification, and loops internally until complete. Use when given requirements/specs to implement autonomously, or when asked to "use Ralph Wiggums", or mentions "Ralph" at all. Runs on your Claude Code subscription—no API credits needed.
---

# Ralph Wiggums (Subscription Mode)

Autonomous development methodology that runs **entirely within Claude Code**. No external bash loops, no API calls—just continuous internal iteration until all stories complete.

## Core Philosophy

- **Memory is the filesystem** — Each iteration inherits prd.json, progress.txt, git history, and code state
- **Failures are information** — Predictable failures teach what guardrails to add
- **Verification is the product** — Tests aren't optional; they're the feedback loop
- **Learnings compound** — By story 10, patterns from stories 1-9 are in progress.txt
- **Continuous execution** — Loop internally until done, no user input between iterations

## Workflow Overview

```
Phase 1: ANALYSIS      → Read requirements, detect stack, understand scope
Phase 2: DECOMPOSITION → Generate atomic stories with explicit criteria + tests
Phase 3: SCAFFOLDING   → Create scripts/ralph/ with prd.json, prompt.md, progress.txt
Phase 4: EXECUTION     → Autonomous internal loop until <promise>COMPLETE</promise>
```

---

## AUTONOMOUS EXECUTION LOOP

**CRITICAL: This is the core of Ralph Wiggums. Once started, continue looping without user input.**

When the user says "Use Ralph Wiggums" or "Continue Ralph Wiggums", or mentions "Ralph" at all, enter this loop:

### The Loop (repeat until done)

```
┌─────────────────────────────────────────────────────────────┐
│  1. READ STATE                                              │
│     - Read scripts/ralph/prd.json                           │
│     - Read scripts/ralph/progress.txt (patterns section)    │
│     - Check git status                                      │
│                                                             │
│  2. SELECT STORY                                            │
│     - Find highest priority story where passes=false        │
│     - If ALL stories pass → output <promise>COMPLETE</promise> and STOP │
│                                                             │
│  3. IMPLEMENT                                               │
│     - Implement ONE story only                              │
│     - Follow patterns from progress.txt                     │
│     - Write tests during implementation                     │
│                                                             │
│  4. VERIFY                                                  │
│     - Run typecheck (ruff check . or tsc --noEmit)          │
│     - Run unit tests (pytest or npm test)                   │
│     - Run lint                                              │
│     - If visual: take screenshot                            │
│                                                             │
│  5. RECORD                                                  │
│     - If passing: commit, update prd.json passes=true       │
│     - Append to progress.txt with learnings                 │
│     - Add patterns to Codebase Patterns section             │
│                                                             │
│  6. CONTINUE IMMEDIATELY                                    │
│     ─────────────────────────────────────────────────────── │
│     │  ⚠️  DO NOT WAIT FOR USER INPUT                      │ │
│     │  ⚠️  IMMEDIATELY GO BACK TO STEP 1                   │ │
│     ─────────────────────────────────────────────────────── │
│                                                             │
│  EXIT CONDITIONS:                                           │
│     - ALL stories pass → <promise>COMPLETE</promise>        │
│     - Stuck after 5 attempts on same story → <promise>STUCK</promise> │
└─────────────────────────────────────────────────────────────┘
```

### Key Rule

**DO NOT** end your turn after completing a story. **IMMEDIATELY** continue to the next story. The loop runs continuously within this Claude Code session until all work is complete.

---

## Phase 1: Analysis

When given requirements:

1. Read the input requirements file completely
2. Detect project stack:
   - Python: Look for requirements.txt, pyproject.toml → use ruff, pytest
   - TypeScript: Look for tsconfig.json → use tsc, vitest/jest
   - Go: Look for go.mod → use go vet, go test
3. Identify:
   - Core functionality required
   - UI vs backend vs full-stack scope
   - Testing framework in use
   - External dependencies needed

## Phase 2: Decomposition

**Critical**: Stories must be atomic—completable in ONE iteration.

For each feature in requirements, generate stories:

```json
{
  "id": "US-001",
  "title": "Short imperative title",
  "acceptanceCriteria": [
    "Specific observable outcome 1",
    "Specific observable outcome 2",
    "typecheck passes",
    "unit tests pass"
  ],
  "testRequirements": {
    "unit": ["test_function_does_x"],
    "integration": [],
    "visual": false
  },
  "priority": 1,
  "passes": false,
  "notes": ""
}
```

### Story Size Guidelines

❌ **Too big**: "Build entire auth system"
✅ **Right size**:
- "Add login form with email/password fields"
- "Add email format validation"  
- "Add auth server action"
- "Add session persistence"

### Acceptance Criteria Rules

❌ **Vague**: "Users can log in"
✅ **Explicit**:
- Email and password input fields render
- Email format validation shows error for invalid emails
- Submit button disabled when fields empty
- Successful login redirects to /dashboard
- typecheck passes
- unit tests pass

## Phase 3: Scaffolding

Create `scripts/ralph/` directory with:

### 1. prd.json
```json
{
  "projectName": "Project Name",
  "branchName": "ralph/feature-name",
  "stack": {
    "language": "python",
    "framework": "textual",
    "testRunner": "pytest"
  },
  "verification": {
    "typecheck": "ruff check .",
    "unitTest": "pytest tests/ -v",
    "lint": "ruff check ."
  },
  "userStories": [
    // Generated stories here
  ]
}
```

### 2. progress.txt
```markdown
# Ralph Progress Log
Started: [DATE]
Project: [NAME]

## Codebase Patterns
- [Initial patterns discovered during analysis]

## Key Files
- [Critical files identified]

---
```

## Phase 4: Autonomous Execution

Once scaffolding is complete, **immediately enter the autonomous loop**:

1. Read prd.json → find first story where passes=false
2. Implement it
3. Verify it
4. Record results
5. **IMMEDIATELY continue to step 1** (no user input)
6. Repeat until ALL stories pass

---

## Failure Recovery

### Story Fails 3+ Times
1. Log failure pattern in progress.txt
2. Decompose into 2-3 smaller sub-stories
3. Insert sub-stories into prd.json with higher priority
4. Continue with first sub-story

### Stuck (No Progress After 5 Iterations on Same Story)
1. Output `<promise>STUCK</promise>`
2. Document in progress.txt:
   - What was attempted
   - What's blocking
   - Suggested alternatives
3. Stop execution for human review

---

## Verification Commands by Stack

### Python
```bash
ruff check .                    # Lint + typecheck
pytest tests/ -v                # Unit tests
pytest tests/integration/ -v    # Integration tests
```

### TypeScript
```bash
npm run typecheck               # or tsc --noEmit
npm test                        # Unit tests
npm run test:e2e                # E2E tests
npm run lint                    # Linting
```

---

## How to Invoke

### Start Fresh
```
Use the Ralph Wiggums skill to implement [requirements document or description]
```

### Resume Existing Work
```
Continue Ralph Wiggums. Read scripts/ralph/prd.json and progress.txt, then implement remaining stories autonomously.
```

### Check Status
```bash
./scripts/ralph/ralph.sh
```

---

## Quick Reference

| File | Purpose |
|------|---------|
| `scripts/ralph/prd.json` | Task list with stories, criteria, status |
| `scripts/ralph/progress.txt` | Session memory, patterns, learnings |
| `.claude/skills/ralph-wiggums/` | Skill definition and templates |

---

## When NOT to Use Ralph

- Exploratory work (unclear end state)
- Major refactors without clear criteria
- Security-critical code (auth, payments)
- Tasks requiring human judgment/design decisions
- One-shot operations

---

## Additional Resources

- `references/stack-detection.md` — Auto-detect project stack
- `references/story-decomposition.md` — How to break down requirements
- `references/test-strategies.md` — Testing patterns per stack
- `references/failure-recovery.md` — Detailed recovery protocols

