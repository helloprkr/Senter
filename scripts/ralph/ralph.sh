#!/bin/bash
# ralph.sh - The autonomous development loop
# Usage: ./ralph.sh [max_iterations]
# Default: 25 iterations

set -e

MAX_ITERATIONS=${1:-25}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAUSE_FILE="$SCRIPT_DIR/.ralph-pause"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
PROMPT_FILE="$SCRIPT_DIR/prompt.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for required files
check_setup() {
  local missing=0
  
  if [[ ! -f "$PRD_FILE" ]]; then
    echo -e "${RED}❌ Missing: $PRD_FILE${NC}"
    missing=1
  fi
  
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo -e "${RED}❌ Missing: $PROMPT_FILE${NC}"
    missing=1
  fi
  
  if [[ ! -f "$PROGRESS_FILE" ]]; then
    echo -e "${YELLOW}⚠️  Creating: $PROGRESS_FILE${NC}"
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$PROGRESS_FILE"
    echo "" >> "$PROGRESS_FILE"
    echo "## Codebase Patterns" >> "$PROGRESS_FILE"
    echo "- (patterns will be added as discovered)" >> "$PROGRESS_FILE"
    echo "" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
  
  if [[ $missing -eq 1 ]]; then
    echo -e "${RED}Setup incomplete. Run init-ralph.py first.${NC}"
    exit 1
  fi
}

# Check for pause sentinel
check_pause() {
  if [[ -f "$PAUSE_FILE" ]]; then
    echo -e "${YELLOW}⏸️  Pause requested.${NC}"
    echo "Remove $PAUSE_FILE to continue."
    exit 0
  fi
}

# Get current story status
get_status() {
  if command -v jq &> /dev/null; then
    local total=$(jq '.userStories | length' "$PRD_FILE")
    local done=$(jq '[.userStories[] | select(.passes == true)] | length' "$PRD_FILE")
    local blocked=$(jq '[.userStories[] | select(.blocked == true)] | length' "$PRD_FILE" 2>/dev/null || echo "0")
    echo "Progress: $done/$total complete"
    if [[ "$blocked" != "0" ]]; then
      echo -e "${YELLOW}Blocked: $blocked stories${NC}"
    fi
  fi
}

# Main loop
main() {
  echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║       🚀 Starting Ralph Wiggums Loop       ║${NC}"
  echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
  echo ""
  echo "Max iterations: $MAX_ITERATIONS"
  echo "Pause file: $PAUSE_FILE"
  echo ""
  
  check_setup
  get_status
  echo ""
  
  for i in $(seq 1 $MAX_ITERATIONS); do
    check_pause
    
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}═══ Iteration $i of $MAX_ITERATIONS ═══${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo ""
    
    # Run Claude with the prompt
    OUTPUT=$(cat "$PROMPT_FILE" \
      | claude --dangerously-skip-permissions 2>&1 \
      | tee /dev/stderr) || true
    
    # Check for completion
    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
      echo ""
      echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
      echo -e "${GREEN}║         ✅ All stories complete!           ║${NC}"
      echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
      get_status
      exit 0
    fi
    
    # Check for stuck signal
    if echo "$OUTPUT" | grep -q "<promise>STUCK</promise>"; then
      echo ""
      echo -e "${YELLOW}╔═══════════════════════════════════════════╗${NC}"
      echo -e "${YELLOW}║   🔄 Story stuck - review progress.txt    ║${NC}"
      echo -e "${YELLOW}╚═══════════════════════════════════════════╝${NC}"
      echo "Continuing with remaining stories..."
    fi
    
    # Brief pause between iterations
    echo ""
    get_status
    echo "Waiting 2 seconds before next iteration..."
    sleep 2
  done
  
  echo ""
  echo -e "${YELLOW}╔═══════════════════════════════════════════╗${NC}"
  echo -e "${YELLOW}║  ⚠️  Max iterations ($MAX_ITERATIONS) reached  ║${NC}"
  echo -e "${YELLOW}╚═══════════════════════════════════════════╝${NC}"
  echo ""
  get_status
  echo ""
  echo "Review: $PROGRESS_FILE"
  exit 1
}

main
