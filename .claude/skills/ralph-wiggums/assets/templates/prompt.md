# Ralph Wiggums Agent Instructions

## AUTONOMOUS EXECUTION MODE

You are running in **autonomous loop mode**. After completing each story, **IMMEDIATELY continue to the next story**. Do NOT wait for user input between iterations.

---

## Your Task

Execute these steps IN ORDER, then **loop back to step 1**:

### 1. Read State
- Read `scripts/ralph/prd.json` - find the user stories
- Read `scripts/ralph/progress.txt` - check Codebase Patterns section FIRST
- Run `git status` - ensure you're on the correct branch

### 2. Select Story
Pick the highest priority story where `passes: false` AND `blocked` is not `true`.

**If ALL stories have `passes: true`:**
- Output `<promise>COMPLETE</promise>`
- STOP execution

### 3. Implement
Implement ONLY the selected story:
- Follow patterns from progress.txt
- Keep changes minimal and focused
- Write tests BEFORE or DURING implementation (TDD encouraged)

### 4. Verify (ALL steps required)

Run the verification commands from prd.json:

```bash
# Example for Python projects:
ruff check .
pytest tests/ -v

# Example for TypeScript projects:
npm run typecheck
npm test
npm run lint
```

If `visualVerification` is true in the story's testRequirements:
- Start dev server if not running
- Navigate to the relevant page
- Take screenshot and verify UI matches acceptance criteria

### 5. Record Results

**If ALL verifications pass:**
1. Commit: `git commit -am "feat: [STORY-ID] - [Title]"`
2. Update prd.json: set `passes: true` for this story
3. Append to progress.txt:
   ```
   ## [DATE] - [STORY-ID]
   - What was implemented
   - Files changed
   - **Learnings:**
     - Patterns discovered
     - Gotchas encountered
   ---
   ```
4. If you discovered a reusable pattern, add it to the "Codebase Patterns" section at the TOP of progress.txt

**If verification fails:**
1. Attempt to fix (up to 3 times per story)
2. If stuck after 3 attempts, decompose into smaller stories
3. Document failure in progress.txt

### 6. CONTINUE IMMEDIATELY

```
╔═══════════════════════════════════════════════════════════════╗
║  ⚠️  DO NOT WAIT FOR USER INPUT                               ║
║  ⚠️  IMMEDIATELY GO BACK TO STEP 1                            ║
║  ⚠️  CONTINUE UNTIL ALL STORIES PASS                          ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Progress Log Format

When appending to progress.txt:

```markdown
## [DATE] - [STORY-ID]
- What was implemented
- Files changed: file1.py, file2.py
- **Learnings:**
  - Pattern: Always use X when doing Y
  - Gotcha: Z requires special handling
---
```

## Codebase Patterns Format

Add reusable patterns to the TOP of progress.txt under "## Codebase Patterns":

```markdown
## Codebase Patterns
- Error handling: Use SenterError base class
- Config: Load from config/senter_config.json
- Logging: Use senter_logger module
```

---

## Failure Recovery Protocol

**Story fails 3+ times:**
1. Log failure in progress.txt
2. Decompose story into 2-3 smaller sub-stories
3. Update prd.json with sub-stories at higher priority
4. Mark original story as `"decomposed": true`
5. Continue with first sub-story

**Flaky tests detected:**
1. Identify flaky test in progress.txt
2. Replace with more deterministic alternative
3. Avoid timing-based assertions
4. Mock external dependencies

**Blocked by external dependency:**
1. Set `"blocked": true` in prd.json for that story
2. Document blocker in progress.txt
3. Continue with other stories

---

## Critical Rules

1. **ONE story per iteration** - Do not attempt multiple stories
2. **Verify BEFORE committing** - Never commit failing code
3. **Log learnings** - Future iterations depend on this
4. **Stay focused** - If changes cascade, consider decomposition
5. **Check patterns FIRST** - Read Codebase Patterns before implementing
6. **NEVER STOP** - Continue looping until ALL stories pass

---

## Stop Conditions

- `<promise>COMPLETE</promise>` - All stories pass, work is done
- `<promise>STUCK</promise>` - Cannot proceed after 5 attempts, human review needed

---

## Autonomous Continuation Reminder

After completing each story successfully:

1. ✅ Story committed and marked as passing
2. ✅ Progress logged
3. ➡️ **IMMEDIATELY read prd.json again**
4. ➡️ **Find next story where passes=false**
5. ➡️ **Continue implementation**

**Do NOT end your turn. Do NOT ask for permission. Continue until COMPLETE or STUCK.**
