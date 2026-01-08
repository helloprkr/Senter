# Ralph Wiggums - Autonomous Iteration Protocol

**You are executing the Ralph Wiggums autonomous development loop.**
**Continue iterating until ALL stories pass. Do NOT wait for user input.**

---

## LOOP INSTRUCTION

Execute these steps repeatedly until complete:

### 1. Read State
```bash
cat scripts/ralph/prd.json      # Stories and status
cat scripts/ralph/progress.txt  # Patterns (read this FIRST!)
git status                      # Verify branch
```

### 2. Select Story
- Pick highest priority story where `passes: false` AND `blocked` is not `true`
- If ALL stories have `passes: true` → output `<promise>COMPLETE</promise>` and STOP

### 3. Implement ONE Story
- Follow patterns from "Codebase Patterns" in progress.txt
- Write tests BEFORE or DURING implementation
- Keep changes focused on this ONE story only

### 4. Verify (ALL must pass)

```bash
# Typecheck
{{TYPECHECK_COMMAND}}

# Unit tests
{{UNIT_TEST_COMMAND}}

# Integration tests (if applicable)
{{INTEGRATION_TEST_COMMAND}}

# Lint
{{LINT_COMMAND}}
```

If `visual: true` in testRequirements → verify UI visually

### 5. On Success: Commit & Record

```bash
git add -A
git commit -m "feat: [US-XXX] - [Title]"
```

Update `scripts/ralph/prd.json`:
- Set `passes: true` for this story

Append to `scripts/ralph/progress.txt`:
```markdown
## [DATE] - US-XXX: [Title]
- Implemented: [what]
- Files: [list]
- **Learnings:** [patterns discovered]
---
```

Add reusable patterns to "## Codebase Patterns" at TOP of progress.txt.

### 6. On Failure: Recovery

1. Attempt fix (up to 3 times)
2. If stuck after 3 attempts:
   - Decompose into 2-3 smaller sub-stories
   - Update prd.json with sub-stories
   - Set `decomposed: true` on original
   - Log to progress.txt
3. Continue with first sub-story

### 7. CONTINUE IMMEDIATELY

**After steps 1-6, GO BACK TO STEP 1.**
- Do NOT wait for user input
- Do NOT ask for confirmation
- Continue until all stories pass

---

## Exit Conditions

| Condition | Action |
|-----------|--------|
| All stories `passes: true` | Output `<promise>COMPLETE</promise>`, STOP |
| Cannot proceed, blocked | Output `<promise>STUCK</promise>`, STOP |

---

## Critical Rules

1. **ONE story per iteration** — never multiple
2. **VERIFY before committing** — never commit failing code
3. **LOG learnings** — progress.txt is memory
4. **DECOMPOSE when stuck** — 3 failures = split
5. **CONTINUE AUTONOMOUSLY** — no waiting for user

---

**BEGIN ITERATION NOW.**
