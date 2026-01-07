# How It Works

A bash loop that:

1. Pipes a prompt into your AI agent
2. Agent picks the next story from prd.json
3. Agent implements it
4. Agent runs typecheck + tests
5. Agent commits if passing
6. Agent marks story done
7. Agent logs learnings
8. Loop repeats until done
Memory persists only through:
scripts/ralph/
├── ralph.sh
├── prompt.md
├── prd.json
└── progress.txt
ralph.sh
The loop:
bash
#!/bin/bash
set -e

MAX_ITERATIONS=${1:-10}
SCRIPT_DIR="$(cd "$(dirname \
  "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting Ralph"

for i in $(seq 1 $MAX_ITERATIONS); do
  echo "═══ Iteration $i ═══"
  
  OUTPUT=$(cat "$SCRIPT_DIR/prompt.md" \
    | amp --dangerously-allow-all 2>&1 \
    | tee /dev/stderr) || true
  
  if echo "$OUTPUT" | \
    grep -q "<promise>COMPLETE</promise>"
  then
    echo "✅ Done!"
    exit 0
  fi
  
  sleep 2
done

echo "⚠️ Max iterations reached"
exit 1
Make executable:
bash
chmod +x scripts/ralph/ralph.sh
Other agents:
Claude Code: `claude --dangerously-skip-permissions`
prompt.md
Instructions for each iteration:
markdown
# Ralph Agent Instructions

## Your Task

1. Read `scripts/ralph/prd.json`
2. Read `scripts/ralph/progress.txt`
   (check Codebase Patterns first)
3. Check you're on the correct branch
4. Pick highest priority story 
   where `passes: false`
5. Implement that ONE story
6. Run typecheck and tests
7. Update AGENTS.md files with learnings
8. Commit: `feat: [ID] - [Title]`
9. Update prd.json: `passes: true`
10. Append learnings to progress.txt

## Progress Format

APPEND to progress.txt:

## [Date] - [Story ID]
- What was implemented
- Files changed
- **Learnings:**
  - Patterns discovered
  - Gotchas encountered
---

## Codebase Patterns

Add reusable patterns to the TOP 
of progress.txt:

## Codebase Patterns
- Migrations: Use IF NOT EXISTS
- React: useRef<Timeout | null>(null)

## Stop Condition

If ALL stories pass, reply:
<promise>COMPLETE</promise>

Otherwise end normally.
prd.json
Your task list:
json
{
  "branchName": "ralph/feature",
  "userStories": [
    {
      "id": "US-001",
      "title": "Add login form",
      "acceptanceCriteria": [
        "Email/password fields",
        "Validates email format",
        "typecheck passes"
      ],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
Key fields:
`branchName` — branch to use
`priority` — lower = first
`passes` — set true when done
progress.txt
Start with context:
markdown
# Ralph Progress Log
Started: 2024-01-15

## Codebase Patterns
- Migrations: IF NOT EXISTS
- Types: Export from actions.ts

## Key Files
- db/schema.ts
- app/auth/actions.ts
---
Ralph appends after each story.
Patterns accumulate across iterations.
Running Ralph
bash
./scripts/ralph/ralph.sh 25
Runs up to 25 iterations.
Ralph will:
Create the feature branch
Complete stories one by one
Commit after each
Stop when all pass
Critical Success Factors
1. Small Stories
Must fit in one context window.
plaintext
❌ Too big:
> "Build entire auth system"
✅ Right size:
> "Add login form"
> "Add email validation"
> "Add auth server action"
2. Feedback Loops
Ralph needs fast feedback:
`npm run typecheck`
`npm test`
Without these, broken code compounds.
3. Explicit Criteria
plaintext
❌ Vague:
> "Users can log in"
✅ Explicit:
> - Email/password fields
> - Validates email format
> - Shows error on failure
> - typecheck passes
> - Verify at localhost:$PORT/login (PORT defaults to 3000)
4. Learnings Compound
By story 10, Ralph knows patterns from stories 1-9.
Two places for learnings:
progress.txt — session memory for Ralph iterations
AGENTS.md — permanent docs for humans and future agents
Before committing, Ralph updates AGENTS.md files in directories with edited files if it discovered reusable patterns (gotchas, conventions, dependencies).
5. AGENTS.md Updates
Ralph updates AGENTS.md when it learns something worth preserving:
plaintext
✅ Good additions:
- "When modifying X, also update Y"
- "This module uses pattern Z"
- "Tests require dev server running"
❌ Don't add:
- Story-specific details
- Temporary notes
- Info already in progress.txt
6. Browser Testing
For UI changes, use the dev-browser skill by @sawyerhood. Load it with `Load the dev-browser skill`, then:
bash
# Start the browser server
~/.config/amp/skills/dev-browser/server.sh &
# Wait for "Ready" message

# Write scripts using heredocs
cd ~/.config/amp/skills/dev-browser && npx tsx <<'EOF'
import { connect, waitForPageLoad } from "@/client.js";

const client = await connect();
const page = await client.page("test");
await page.setViewportSize({ width: 1280, height: 900 });
const port = process.env.PORT || "3000";
await page.goto(`http://localhost:${port}/your-page`);
await waitForPageLoad(page);
await page.screenshot({ path: "tmp/screenshot.png" });
await client.disconnect();
EOF
Not complete until verified with screenshot.
Common Gotchas
Idempotent migrations:
sql
ADD COLUMN IF NOT EXISTS email TEXT;
Interactive prompts:
bash
echo -e "\n\n\n" | npm run db:generate
Schema changes:
After editing schema, check:
Server actions
UI components
API routes
Fixing related files is OK:
If typecheck requires other changes, make them. Not scope creep.
Monitoring
bash
# Story status
cat scripts/ralph/prd.json | \
jq '.userStories[] | {id, passes}'
# Learnings
cat scripts/ralph/progress.txt
# Commits
git log --oneline -10
Real Results
We built an evaluation system:
13 user stories
~15 iterations
2-5 min each
~1 hour total
Learnings compound. By story 10, Ralph knew our patterns.
When NOT to Use
Exploratory work
Major refactors without criteria
Security-critical code


## Additional Version of Ralph

Ralph Wiggum Plugin
Implementation of the Ralph Wiggum technique for iterative, self-referential AI development loops in Claude Code.

What is Ralph?
Ralph is a development methodology based on continuous AI agent loops. As Geoffrey Huntley describes it: "Ralph is a Bash loop" - a simple while true that repeatedly feeds an AI agent a prompt file, allowing it to iteratively improve its work until completion.

The technique is named after Ralph Wiggum from The Simpsons, embodying the philosophy of persistent iteration despite setbacks.

Core Concept
This plugin implements Ralph using a Stop hook that intercepts Claude's exit attempts:

# You run ONCE:
/ralph-loop "Your task description" --completion-promise "DONE"

# Then Claude Code automatically:
# 1. Works on the task
# 2. Tries to exit
# 3. Stop hook blocks exit
# 4. Stop hook feeds the SAME prompt back
# 5. Repeat until completion
The loop happens inside your current session - you don't need external bash loops. The Stop hook in hooks/stop-hook.sh creates the self-referential feedback loop by blocking normal session exit.

This creates a self-referential feedback loop where:

The prompt never changes between iterations
Claude's previous work persists in files
Each iteration sees modified files and git history
Claude autonomously improves by reading its own past work in files
Quick Start
/ralph-loop "Build a REST API for todos. Requirements: CRUD operations, input validation, tests. Output <promise>COMPLETE</promise> when done." --completion-promise "COMPLETE" --max-iterations 50
Claude will:

Implement the API iteratively
Run tests and see failures
Fix bugs based on test output
Iterate until all requirements met
Output the completion promise when done
Commands
/ralph-loop
Start a Ralph loop in your current session.

Usage:

/ralph-loop "<prompt>" --max-iterations <n> --completion-promise "<text>"
Options:

--max-iterations <n> - Stop after N iterations (default: unlimited)
--completion-promise <text> - Phrase that signals completion
/cancel-ralph
Cancel the active Ralph loop.

Usage:

/cancel-ralph
Prompt Writing Best Practices
1. Clear Completion Criteria
❌ Bad: "Build a todo API and make it good."

✅ Good:

Build a REST API for todos.

When complete:
- All CRUD endpoints working
- Input validation in place
- Tests passing (coverage > 80%)
- README with API docs
- Output: <promise>COMPLETE</promise>
2. Incremental Goals
❌ Bad: "Create a complete e-commerce platform."

✅ Good:

Phase 1: User authentication (JWT, tests)
Phase 2: Product catalog (list/search, tests)
Phase 3: Shopping cart (add/remove, tests)

Output <promise>COMPLETE</promise> when all phases done.
3. Self-Correction
❌ Bad: "Write code for feature X."

✅ Good:

Implement feature X following TDD:
1. Write failing tests
2. Implement feature
3. Run tests
4. If any fail, debug and fix
5. Refactor if needed
6. Repeat until all green
7. Output: <promise>COMPLETE</promise>
4. Escape Hatches
Always use --max-iterations as a safety net to prevent infinite loops on impossible tasks:

# Recommended: Always set a reasonable iteration limit
/ralph-loop "Try to implement feature X" --max-iterations 20

# In your prompt, include what to do if stuck:
# "After 15 iterations, if not complete:
#  - Document what's blocking progress
#  - List what was attempted
#  - Suggest alternative approaches"
Note: The --completion-promise uses exact string matching, so you cannot use it for multiple completion conditions (like "SUCCESS" vs "BLOCKED"). Always rely on --max-iterations as your primary safety mechanism.

Philosophy
Ralph embodies several key principles:

1. Iteration > Perfection
Don't aim for perfect on first try. Let the loop refine the work.

2. Failures Are Data
"Deterministically bad" means failures are predictable and informative. Use them to tune prompts.

3. Operator Skill Matters
Success depends on writing good prompts, not just having a good model.

4. Persistence Wins
Keep trying until success. The loop handles retry logic automatically.

When to Use Ralph
Good for:

Well-defined tasks with clear success criteria
Tasks requiring iteration and refinement (e.g., getting tests to pass)
Greenfield projects where you can walk away
Tasks with automatic verification (tests, linters)
Not good for:

Tasks requiring human judgment or design decisions
One-shot operations
Tasks with unclear success criteria
Production debugging (use targeted debugging instead)
Real-World Results
Successfully generated 6 repositories overnight in Y Combinator hackathon testing
One $50k contract completed for $297 in API costs
Created entire programming language ("cursed") over 3 months using this approach
Learn More
Original technique: https://ghuntley.com/ralph/
Ralph Orchestrator: https://github.com/mikeyobrien/ralph-orchestrator
For Help
Run /help in Claude Code for detailed command reference and examples.

### Version 3 of Ralph

The Ralph Wiggum Loop: Running AI Agents for Hours, Not Minutes
Published on January 6, 2025
4 min read
#Claude Code
#Autonomous Agents
#Developer Productivity
#AI Engineering
TL;DR
The Ralph Wiggum Loop is a technique where you feed Claude Code the same prompt repeatedly. Each iteration, Claude sees its previous work in files and git history, self-corrects, and makes incremental progress. The result: Complex tasks run autonomously for hours or days - while you sleep.

The Problem: Context is Finite
Anyone working with AI coding assistants knows the feeling: You’re in the middle of a complex refactor, Claude finally understands the context, and then - Context window exhausted. Restart. Explain everything again.

Geoffrey Huntley faced exactly this problem. His solution? A bash loop so simple it’s almost audacious:

while :; do
  cat PROMPT.md | claude-code --continue
done
The name? A tribute to Ralph Wiggum from The Simpsons - someone who just keeps going despite all odds.

How It Works
The magic lies in iteration with memory:

Prompt is stored - Your task lands in a state file
Claude works - Reads files, makes changes, commits
Claude tries to stop - Thinks it’s done
Hook intercepts - Checks for a “promise tag” like <promise>DONE</promise>
Not found? - Prompt gets re-injected, next iteration starts
Each iteration, Claude sees:

The modified files from previous runs
The git history with commit messages
The todo list with remaining tasks
Claude corrects its own mistakes, continues where it left off, and converges step by step toward the solution.

Installation in 30 Seconds
Ralph Wiggum is an official Anthropic plugin:

/plugin install ralph-wiggum@claude-plugins-official
Start a loop:

/ralph-loop "Migrate all tests from Jest to Vitest.
Output <promise>MIGRATION COMPLETE</promise> when done." \
  --max-iterations 30
Cancel anytime with:

/cancel-ralph
When Ralph Shines
Perfect for:

Use Case	Why Ralph Works
Framework migrations	Hundreds of files, same process
Increasing test coverage	TDD loop until all tests green
API documentation	Iteratively document all endpoints
Code standardization	Linting fixes across entire codebase
Real-world example: Geoffrey Huntley ran a loop for 3 months. The result: Cursed - a complete programming language with compiler, standard library, and editor support. Autonomously developed.

An MVP for a $50,000 contract was delivered for $297 in API costs.

When Ralph is the Wrong Choice
Not every task benefits from autonomy:

Architectural decisions - Requires human judgment
Unclear requirements - No “done” criteria means no convergence
Security-critical code - Authentication, payments
Exploratory work - When you don’t know what you want yet
Best Practices
1. Define Clear Completion Criteria
❌ Bad:
"Build a todo API and make it good."

✅ Good:
"Build a REST API for todos.
- CRUD endpoints for /todos
- Input validation
- Error handling
- Tests with >80% coverage

Output <promise>COMPLETE</promise> when ALL requirements met."
2. Always Set max-iterations
# Safety net against infinite loops
/ralph-loop "..." --max-iterations 30
3. Build in Escape Hatches
After 20 iterations without progress:
- Document what's blocking
- List attempted approaches
- Suggest alternatives
- Output <promise>NEEDS HELP</promise>
4. Use a Git Directory
Every iteration auto-commits. If things go wrong: git reset --hard and restart is often faster than debugging.

The Cost Question
Transparency matters:

API costs: 50 iterations on a large codebase = $50-100+
Claude Code subscription: Hits usage limits faster
Time vs. money: A developer day costs more than a $50 loop
The math works out when the alternative is manual work spanning days.

The Mindset Shift
Ralph changes how you work with AI:

Old: Guide Claude step by step New: Design prompts that converge toward correct solutions

Your role becomes prompt architect. You define success, Claude finds the path.

As Huntley puts it:

“Ralph will test you. Every time Ralph has taken a wrong direction, I haven’t blamed the tools - I’ve looked inside.”

Conclusion
The Ralph Wiggum Loop isn’t a silver bullet. But for the right tasks - clearly defined, iterative, time-intensive - it’s a game changer.

The technique shows where we’re heading: From AI as assistant to AI as autonomous colleague, completing tasks overnight while you sleep.

And when you wake up in the morning, the PR is ready for review.

Further Reading
Original article by Geoffrey Huntley
A Brief History of Ralph - HumanLayer
Official plugin on GitHub