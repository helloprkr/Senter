#!/bin/bash
#
# Ralph Wiggums - Autonomous Development Loop
#
# Usage: ./ralph.sh [max_iterations]
#
# This script runs Claude Code in a loop, executing one user story per iteration.
# Create .ralph-pause to pause, delete to resume.
# Ctrl+C to abort.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MAX_ITERATIONS=${1:-50}
ITERATION=0
PAUSE_FILE="$SCRIPT_DIR/.ralph-pause"

echo "======================================"
echo "  Ralph Wiggums Autonomous Loop"
echo "======================================"
echo "Project: $PROJECT_ROOT"
echo "Max iterations: $MAX_ITERATIONS"
echo ""
echo "Controls:"
echo "  - Create $PAUSE_FILE to pause"
echo "  - Delete $PAUSE_FILE to resume"
echo "  - Ctrl+C to abort"
echo ""

cd "$PROJECT_ROOT"

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))

    echo ""
    echo "======================================"
    echo "  Iteration $ITERATION / $MAX_ITERATIONS"
    echo "======================================"
    echo ""

    # Check for pause
    if [ -f "$PAUSE_FILE" ]; then
        echo "[PAUSED] Remove $PAUSE_FILE to continue..."
        while [ -f "$PAUSE_FILE" ]; do
            sleep 5
        done
        echo "[RESUMED]"
    fi

    # Run Claude Code with the prompt
    echo "[Starting iteration...]"

    # Execute Claude Code with the iteration prompt
    # The prompt instructs Claude to:
    # 1. Read prd.json and progress.txt
    # 2. Pick next uncompleted story
    # 3. Implement it
    # 4. Verify
    # 5. Commit and update status

    OUTPUT=$(claude --print "Execute one Ralph Wiggums iteration. Read scripts/ralph/prompt.md for instructions. Read scripts/ralph/prd.json for stories. Read scripts/ralph/progress.txt for patterns. Implement the highest priority story where passes=false. Verify all checks pass. Commit and update prd.json." 2>&1) || true

    echo "$OUTPUT"

    # Check for completion signal
    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
        echo ""
        echo "======================================"
        echo "  ALL STORIES COMPLETE!"
        echo "======================================"
        exit 0
    fi

    # Check for stuck signal
    if echo "$OUTPUT" | grep -q "<promise>STUCK</promise>"; then
        echo ""
        echo "======================================"
        echo "  STUCK - Manual intervention needed"
        echo "======================================"
        exit 1
    fi

    # Brief pause between iterations
    sleep 2
done

echo ""
echo "======================================"
echo "  Max iterations reached ($MAX_ITERATIONS)"
echo "======================================"
