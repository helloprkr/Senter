#!/bin/bash
# ralph.sh - Status helper for Claude Code Ralph Wiggums
#
# Ralph Wiggums now runs ENTIRELY within Claude Code (subscription).
# This script just shows status and usage instructions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_RALPH_DIR="scripts/ralph"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Ralph Wiggums - Status & Usage                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for project ralph directory
if [[ -d "$PROJECT_RALPH_DIR" ]] && [[ -f "$PROJECT_RALPH_DIR/prd.json" ]]; then
  echo -e "${GREEN}✅ Ralph is initialized${NC}"
  echo ""
  echo -e "${CYAN}Stories:${NC}"
  
  # Show story status using Python (works cross-platform)
  python3 -c "
import json
import sys

try:
    with open('$PROJECT_RALPH_DIR/prd.json', 'r') as f:
        data = json.load(f)
    
    stories = data.get('userStories', [])
    total = len(stories)
    done = sum(1 for s in stories if s.get('passes'))
    blocked = sum(1 for s in stories if s.get('blocked'))
    
    for s in stories:
        if s.get('passes'):
            status = '✅'
        elif s.get('blocked'):
            status = '🚫'
        else:
            status = '☐ '
        print(f\"  {status} {s['id']}: {s['title']}\")
    
    print()
    print(f'Progress: {done}/{total} complete', end='')
    if blocked > 0:
        print(f' ({blocked} blocked)', end='')
    print()
except Exception as e:
    print(f'Error reading prd.json: {e}')
    sys.exit(1)
" 2>/dev/null || echo "  (unable to read prd.json)"
  
  echo ""
else
  echo -e "${YELLOW}⚠️  Ralph not initialized for this project${NC}"
  echo ""
  echo "To initialize, tell Claude Code:"
  echo ""
  echo -e "  ${CYAN}Use Ralph Wiggums to implement [your requirements]${NC}"
  echo ""
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}How to use Ralph Wiggums:${NC}"
echo ""
echo "  📝 Start fresh:"
echo -e "     ${GREEN}Use the Ralph Wiggums skill to implement [requirements]${NC}"
echo ""
echo "  ▶️  Resume work:"
echo -e "     ${GREEN}Continue Ralph Wiggums. Read scripts/ralph/prd.json${NC}"
echo -e "     ${GREEN}and progress.txt, implement remaining stories.${NC}"
echo ""
echo "  📊 Check progress:"
echo -e "     ${GREEN}cat scripts/ralph/progress.txt${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Note: Ralph now runs entirely within Claude Code.${NC}"
echo -e "${YELLOW}No API credits needed—uses your subscription.${NC}"
