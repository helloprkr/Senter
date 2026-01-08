---
name: ralph-wiggums
description: Autonomous iterative development loop. ALWAYS USE when asked to "use Ralph Wiggums", implement requirements autonomously, or build features iteratively. This skill runs ENTIRELY within Claude Code (no API calls) for fully autonomous execution.
---

# Ralph Wiggums v2.0

**User-Value-First Autonomous Development**

> "None of this matters unless the app we're building is phenomenally valuable to our users."

**STOP. Before writing ANY code, you MUST complete the Initialization Protocol below.**

---

## Core Philosophy: Value Over Velocity

Ralph Wiggums v2.0 prioritizes **user experience and value delivery** over test-passing velocity.

**The Old Way (v1.0):**
- Focus: Make tests pass, get through stories fast
- Success metric: All tests green
- Problem: Can build "working" code that delivers zero value

**The New Way (v2.0):**
- Focus: Deliver observable, meaningful value to users
- Success metric: User can DO something valuable they couldn't before
- Each story must answer: "What can the user NOW do that they couldn't before?"

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

### Step 3: Understand the USER
Before generating stories, answer these questions:
- **Who is the user?** (developer, consumer, business user, etc.)
- **What problem are they trying to solve?**
- **What does "phenomenal value" look like for them?**
- **What would make them say "WOW, this is amazing"?**

### Step 4: Generate prd.json (User-Value-First Format)

```json
{
  "projectName": "[NAME]",
  "branchName": "ralph/[feature-name]",
  "userProfile": {
    "who": "Description of target user",
    "coreProblem": "What problem they're solving",
    "wowMoment": "What would make them say WOW"
  },
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
  "userStories": []
}
```

### Step 5: Decompose Requirements into VALUE-DRIVEN Stories

**THE CRITICAL DIFFERENCE: Stories are about USER VALUE, not code tasks.**

❌ **BAD (Code-Task Stories):**
- "Add HTTP endpoint for goals"
- "Create database schema"
- "Implement API client"
- "Add unit tests for X"

✅ **GOOD (User-Value Stories):**
- "User can see their detected goals when they open the app"
- "User receives a 'welcome back' summary after being away"
- "User gets proactive suggestions that feel relevant and helpful"
- "User can have a conversation that Senter remembers next time"

**Story Format (v2.0):**
```json
{
  "id": "US-001",
  "title": "[User can/gets/sees] [specific value]",
  "userValue": "What the user can NOW do that they couldn't before",
  "wowFactor": "What makes this feel magical, not just functional",
  "acceptanceCriteria": [
    "Observable user outcome 1",
    "Observable user outcome 2",
    "Feels smooth/fast/delightful (not just 'works')"
  ],
  "technicalRequirements": {
    "tests": ["test cases needed"],
    "performance": "latency/speed requirements",
    "polish": "UI/UX details that matter"
  },
  "priority": 1,
  "valueDelivered": false,
  "notes": ""
}
```

**Decomposition Questions:**
1. Can a user SEE or EXPERIENCE this value immediately?
2. Would a demo of this make someone say "that's cool"?
3. Is this a complete "unit of value" (not a partial implementation)?

### Step 6: Generate progress.txt
Create `scripts/ralph/progress.txt`:

```markdown
# Ralph Progress Log
Started: [CURRENT DATE]
Project: [PROJECT NAME]

## User Value Focus
Target User: [who]
Core Problem: [what]
WOW Moment: [what would amaze them]

## Value Delivered So Far
- (List user-visible improvements)

## Codebase Patterns
- Stack: [language]/[framework]
- (Add patterns as discovered)

## Key Files
- (List critical files)

---
```

### Step 7: Create Git Branch
```bash
git checkout -b ralph/[feature-name]
```

### Step 8: Begin Autonomous Execution
**IMMEDIATELY after initialization, begin the autonomous loop below.**

---

## 🔄 AUTONOMOUS EXECUTION LOOP (Value-First)

```
┌─────────────────────────────────────────────────────────────┐
│  RALPH WIGGUMS v2.0 - USER VALUE LOOP                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LOOP:                                                      │
│    1. Read scripts/ralph/prd.json                          │
│    2. Read scripts/ralph/progress.txt                      │
│    3. Find highest priority story where valueDelivered=false│
│    4. If NO incomplete stories → COMPLETE, stop            │
│    5. Implement the ONE selected story                     │
│    6. VERIFY VALUE (not just tests!)                       │
│       a. Technical: typecheck, tests, lint                 │
│       b. VALUE CHECK: Can user SEE/USE this improvement?   │
│       c. POLISH CHECK: Does it feel good, not just work?   │
│    7. If VALUE DELIVERED → commit, update, log             │
│    8. If VALUE MISSING → iterate (don't just fix tests)    │
│    9. GOTO step 1                                          │
│                                                             │
│  EXIT CONDITIONS:                                           │
│    - All value delivered → <promise>COMPLETE</promise>     │
│    - Cannot deliver value → <promise>STUCK</promise>       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Value Verification Checklist

Before marking ANY story as `"valueDelivered": true`, confirm:

**Technical (Must Pass):**
- [ ] Typecheck passes
- [ ] Tests pass
- [ ] Lint passes

**Value (THE REAL TEST):**
- [ ] User can SEE or USE this improvement
- [ ] It's a COMPLETE unit of value (not partial)
- [ ] Demo would make someone say "that's useful"

**Polish (What Makes It Great):**
- [ ] Feels fast (no unnecessary delays)
- [ ] Looks good (attention to UI details)
- [ ] Handles edge cases gracefully
- [ ] Error messages are helpful, not cryptic

### On Commit: Value-Focused Message

```bash
git commit -m "$(cat <<'EOF'
feat: [US-XXX] User can [value delivered]

VALUE: [What user can now do]
WOW: [What makes this feel good]

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

### Progress Log Format (Value-Focused)

```markdown
## [DATE] - US-XXX: [User can...]
- **Value Delivered:** [What user can now do]
- **WOW Factor:** [What makes it feel magical]
- **Polish Details:** [UI/UX attention given]
- Files: [changed files]
- Learnings: [patterns discovered]
---
```

---

## Stop Conditions

| Signal | Meaning |
|--------|---------|
| `<promise>COMPLETE</promise>` | All user value delivered. Product is ready. |
| `<promise>STUCK</promise>` | Cannot deliver value. Human review needed. |

---

## Critical Rules (v2.0)

1. **VALUE OVER VELOCITY** — A passing test suite with zero user value = FAILURE
2. **USER-VISIBLE OUTCOMES** — Every story must change what user can SEE or DO
3. **COMPLETE UNITS** — Partial implementations don't count as value
4. **POLISH MATTERS** — "Works" ≠ "Good". Users feel the difference.
5. **DEMO-READY** — Each completed story should be demo-able
6. **AUTONOMOUS BUT THOUGHTFUL** — Don't rush through; quality > speed

---

## Example: Good vs Bad Stories

**Building an AI Assistant:**

❌ BAD Stories (Code-focused):
1. "Create /api/goals endpoint"
2. "Add goal detection class"
3. "Write unit tests for goal detection"
4. "Add frontend API client"
5. "Create goals UI component"

✅ GOOD Stories (Value-focused):
1. "User sees their AI-detected goals when opening the app"
   - Value: User learns what Senter thinks they're working toward
   - WOW: Goals appear without user having to enter them manually
   - Polish: Progress bars, categories, smooth animations

2. "User receives helpful suggestions based on their activity"
   - Value: User gets actionable recommendations
   - WOW: Suggestions feel relevant, not generic
   - Polish: Unobtrusive toasts, easy to act on or dismiss

3. "User gets a 'welcome back' summary after being away"
   - Value: User learns what happened while they were gone
   - WOW: Senter actually did work while they were away
   - Polish: Beautiful modal, clear summary, actionable insights

---

## Quick Reference

| File | Purpose |
|------|---------|
| `scripts/ralph/prd.json` | Stories with VALUE focus |
| `scripts/ralph/progress.txt` | Value delivered, learnings |
| `scripts/ralph/.ralph-pause` | Create to pause |

---

## Execution Summary

When Ralph Wiggums v2.0 is invoked:

1. **Initialize** — Understand user, detect stack, generate VALUE-focused stories
2. **Loop** — Execute until all VALUE is delivered (not just tests pass)
3. **Signal** — Output `<promise>COMPLETE</promise>` when product is valuable

**Remember: We're not writing code. We're delivering value to humans.**
