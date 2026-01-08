#!/bin/bash
#
# Ralph Wiggums - Usage Helper
#
# This script no longer calls the Claude API.
# Ralph Wiggums now runs entirely within Claude Code (subscription-based).
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              🚀 Ralph Wiggums - Autonomous Loop               ║${NC}"
echo -e "${BLUE}╠═══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Runs entirely within Claude Code (NO API, uses subscription) ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in a project with Ralph scaffolding
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/prd.json" ]]; then
    echo -e "${GREEN}✓ Found prd.json${NC}"
    
    if command -v jq &> /dev/null; then
        total=$(jq '.userStories | length' "$SCRIPT_DIR/prd.json")
        done=$(jq '[.userStories[] | select(.passes == true)] | length' "$SCRIPT_DIR/prd.json")
        echo -e "  Progress: ${CYAN}$done/$total${NC} stories complete"
    fi
else
    echo -e "${YELLOW}⚠ No prd.json found - initialization required${NC}"
fi

if [[ -f "$SCRIPT_DIR/progress.txt" ]]; then
    echo -e "${GREEN}✓ Found progress.txt${NC}"
else
    echo -e "${YELLOW}⚠ No progress.txt found${NC}"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    HOW TO USE RALPH WIGGUMS                    ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  1. Open Claude Code in your terminal"
echo ""
echo "  2. Run this command in Claude Code:"
echo ""
echo -e "     ${GREEN}Use the Ralph Wiggums skill to implement [YOUR REQUIREMENTS]${NC}"
echo ""
echo "  3. Claude Code will:"
echo "     • Initialize scaffolding (if needed)"
echo "     • Decompose requirements into stories"
echo "     • Loop through ALL stories autonomously"
echo "     • Commit each completed story"
echo "     • Continue until COMPLETE or STUCK"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Alternatively, to resume an existing project:"
echo ""
echo -e "     ${GREEN}Continue Ralph Wiggums. Read scripts/ralph/prd.json and${NC}"
echo -e "     ${GREEN}progress.txt, then implement remaining stories autonomously.${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Controls:"
echo "    • Pause:  touch scripts/ralph/.ralph-pause"
echo "    • Resume: rm scripts/ralph/.ralph-pause"
echo ""
