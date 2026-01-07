# Ralph Wiggums Agent Instructions

## Your Task

Each iteration, execute these steps IN ORDER:

### 1. Read State
- Read `scripts/ralph/prd.json` - find the user stories
- Read `scripts/ralph/progress.txt` - check Codebase Patterns section FIRST
- Run `git status` - ensure you're on the correct branch

### 2. Select Story
Pick the highest priority story where `passes: false` AND `blocked` is not `true`.
If ALL stories have `passes: true`, output `<promise>COMPLETE</promise>` and stop.

### 3. Implement
Implement ONLY the selected story:
- Follow patterns from progress.txt
- Keep changes minimal and focused
- Write tests BEFORE or DURING implementation (TDD encouraged)

### 4. Verify (ALL steps required)

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
2. If stuck after 3 attempts, decompose into smaller stories (see failure recovery protocol)
3. Document failure in progress.txt

### 6. Continue or Complete

- More stories with `passes: false`? → End this iteration (loop will continue)
- All stories `passes: true`? → Output: `<promise>COMPLETE</promise>`
- Stuck and cannot proceed? → Output: `<promise>STUCK</promise>`

## Progress Log Format

When appending to progress.txt:

```markdown
## [DATE] - [STORY-ID]
- What was implemented
- Files changed: file1.ts, file2.ts
- **Learnings:**
  - Pattern: Always use X when doing Y
  - Gotcha: Z requires special handling
---
```

## Codebase Patterns Format

Add reusable patterns to the TOP of progress.txt under "## Codebase Patterns":

```markdown
## Codebase Patterns
- Migrations: Use IF NOT EXISTS for idempotency
- Forms: Use react-hook-form with zod validation
- API: Always return consistent error shape
```

## Failure Recovery Protocol

**Story fails 3+ times:**
1. Log failure in progress.txt
2. Decompose story into 2-3 smaller sub-stories
3. Update prd.json with sub-stories at higher priority
4. Mark original story as decomposed
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

## Critical Rules

1. **ONE story per iteration** - Do not attempt multiple stories
2. **Verify BEFORE committing** - Never commit failing code
3. **Log learnings** - Future iterations depend on this
4. **Stay focused** - If changes cascade, consider decomposition
5. **Check patterns FIRST** - Read Codebase Patterns before implementing

## Stop Conditions

- `<promise>COMPLETE</promise>` - All stories pass, work is done
- `<promise>STUCK</promise>` - Cannot proceed, human review needed
