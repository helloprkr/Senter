# SENTER: Polish & Perfect

**Current State**: 26/26 tests pass, core features implemented  
**Goal**: Make it actually delightful to use daily  
**Focus**: UX, robustness, configuration, and integration

---

## WHAT'S WORKING VS. WHAT'S ROUGH

| Working | Rough Edges |
|---------|-------------|
| Semantic routing | Warnings on startup clutter the experience |
| Memory save/recall | No feedback when memory is used in response |
| Goal extraction | Goals extracted but not surfaced proactively |
| Learning system | Learned preferences not visibly applied |
| Parallel research | Research results not clearly attributed |
| Web search | Results quality could be better integrated |

---

## PROMPT SET 1: CLEAN UP THE EXPERIENCE

### Prompt 1A: Silence Non-Critical Warnings

```markdown
# Task: Clean Up Startup Warnings

## Problem
When Senter starts, users see warnings about optional models:
- "Omni 3B model not found"
- "Embedding model not found" 
- "TTS model not found"

These are optional features, not errors. They create anxiety and confusion.

## Solution
1. Create a config option `show_optional_warnings: false`
2. Only show warnings for REQUIRED components that are missing
3. For optional components, log to file but don't print to console
4. Add a `--verbose` flag to show all warnings if debugging

## Implementation

1. Update `config/user_profile.json`:
```json
{
  "display": {
    "show_optional_warnings": false,
    "verbose_mode": false
  }
}
```

2. Create a logging wrapper in `Functions/logger.py`:
```python
import logging
from pathlib import Path

def setup_logging(verbose: bool = False):
    """Configure logging - file always, console only for errors/verbose"""
    log_file = Path("data/senter.log")
    log_file.parent.mkdir(exist_ok=True)
    
    # Always log everything to file
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if verbose else logging.NullHandler()
        ]
    )

def optional_warning(message: str):
    """Log optional feature warning - file only unless verbose"""
    logging.debug(f"[OPTIONAL] {message}")

def required_error(message: str):
    """Log required component error - always show"""
    logging.error(message)
    print(f"❌ {message}")
```

3. Replace all optional warnings in codebase with `optional_warning()`

## Acceptance Criteria
- [ ] Clean startup with no scary warnings
- [ ] `--verbose` flag shows all warnings
- [ ] All logs go to data/senter.log
- [ ] Required errors still show prominently
```

---

### Prompt 1B: Show When Features Are Active

```markdown
# Task: Add Visual Feedback for Active Features

## Problem
Users can't tell when memory, goals, or learning are being used.
The system works silently - no feedback on what's happening.

## Solution
Add subtle indicators showing when features activate:

```
[general] You: What's the latest on my project?

[🧠 Memory: Found 3 relevant conversations]
[🎯 Goal: "Finish project by Friday" is active]

Senter: Based on our previous discussions about your project...
```

## Implementation

1. Update response flow in `scripts/senter.py`:
```python
def process_query(query: str) -> str:
    indicators = []
    
    # Check memory
    memory_results = memory.search_memory(query, limit=3)
    if memory_results:
        indicators.append(f"🧠 Memory: Found {len(memory_results)} relevant conversations")
    
    # Check goals
    relevant_goals = goal_tracker.get_relevant_goals(query)
    if relevant_goals:
        indicators.append(f"🎯 Goal: \"{relevant_goals[0].description}\" is active")
    
    # Check if learning applied
    preferences = learner.get_applied_preferences(query)
    if preferences:
        indicators.append(f"📚 Applied: {', '.join(preferences)}")
    
    # Check if research triggered
    if parallel_inference.needs_research(query):
        indicators.append("🔍 Research: Searching in parallel...")
    
    # Show indicators
    for ind in indicators:
        print(f"  [{ind}]")
    
    # Get response with context
    response = generate_response(query, memory_results, relevant_goals)
    return response
```

2. Make indicators optional via config:
```json
{
  "display": {
    "show_feature_indicators": true
  }
}
```

## Acceptance Criteria
- [ ] Memory usage shows indicator with count
- [ ] Active goals show when relevant
- [ ] Learning preferences show when applied
- [ ] Parallel research shows when triggered
- [ ] Can be disabled in config
```

---

### Prompt 1C: Improve Response Attribution

```markdown
# Task: Attribute Information Sources in Responses

## Problem
When Senter uses web search, memory, or goal context, it's not clear where info came from.
Users can't distinguish fresh research from memorized context from LLM knowledge.

## Solution
Add subtle attribution to responses:

```
Senter: Based on our conversation last Tuesday, you mentioned wanting to focus on 
Python. [memory]

The latest Python 3.13 was released in October 2024 with several performance 
improvements. [web: python.org]

For your goal of "learning Python by next month," I'd suggest... [goal context]
```

## Implementation

1. Track sources during response generation:
```python
class ResponseContext:
    def __init__(self):
        self.sources = []
    
    def add_memory(self, conversation_date: str):
        self.sources.append(f"memory: {conversation_date}")
    
    def add_web(self, url: str):
        self.sources.append(f"web: {url}")
    
    def add_goal(self, goal_desc: str):
        self.sources.append(f"goal: {goal_desc}")
```

2. Include sources in LLM prompt:
```python
system_prompt = f"""
You have access to the following context:

MEMORY CONTEXT:
{memory_context}

WEB RESEARCH:
{web_results}

USER GOALS:
{goal_context}

When using information from these sources, naturally mention where it came from.
For example: "Based on what you mentioned before..." or "According to recent news..."
"""
```

3. Optionally show source summary after response:
```
───────────────────────────────
Sources: 2 memories, 1 web result, 1 goal
───────────────────────────────
```

## Acceptance Criteria
- [ ] Responses naturally attribute sources
- [ ] Source summary shown after response (optional)
- [ ] Web results include domain attribution
- [ ] Memory includes approximate date
```

---

## PROMPT SET 2: PROACTIVE INTELLIGENCE

### Prompt 2A: Proactive Goal Surfacing

```markdown
# Task: Proactively Surface Relevant Goals

## Problem
Goals are extracted and stored, but Senter doesn't proactively mention them.
A phenomenal assistant would say "Hey, you mentioned wanting to finish X by Friday - how's that going?"

## Solution
At session start and during relevant conversations, surface active goals:

```
[Senter] Welcome back! You have 2 active goals:
  • "Finish project by Friday" (2 days left)
  • "Learn Python by next month" (24 days left)

Would you like to work on either of these?
```

## Implementation

1. Add goal surfacing to session start in `scripts/senter.py`:
```python
def on_session_start():
    goals = goal_tracker.get_active_goals()
    if goals:
        print(f"\n📋 Active goals ({len(goals)}):")
        for goal in goals[:3]:  # Top 3
            days_left = goal.days_until_deadline()
            urgency = "⚠️" if days_left and days_left < 3 else ""
            print(f"  {urgency} \"{goal.description}\"", end="")
            if days_left:
                print(f" ({days_left} days left)")
            else:
                print()
        print()
```

2. Add deadline tracking to goals:
```python
# In Functions/goal_tracker.py
class Goal:
    def days_until_deadline(self) -> Optional[int]:
        if not self.deadline:
            return None
        delta = self.deadline - datetime.now()
        return max(0, delta.days)
    
    def is_urgent(self) -> bool:
        days = self.days_until_deadline()
        return days is not None and days < 3
```

3. During conversation, check for goal relevance:
```python
def check_goal_relevance(query: str, response: str):
    """After responding, check if we should mention a goal"""
    goals = goal_tracker.get_relevant_goals(query)
    for goal in goals:
        if goal.is_urgent() and goal.id not in mentioned_this_session:
            print(f"\n💡 Reminder: Your goal \"{goal.description}\" is due soon!")
            mentioned_this_session.add(goal.id)
```

## Acceptance Criteria
- [ ] Session start shows active goals with deadlines
- [ ] Urgent goals (< 3 days) get warning icon
- [ ] Relevant goals mentioned during conversation (max once per session)
- [ ] Can ask "what are my goals?" for full list
```

---

### Prompt 2B: Smart Session Summaries

```markdown
# Task: Generate Session Summaries on Exit

## Problem
When user types /exit, Senter just exits. No summary of what was discussed, decided, or learned.

## Solution
On exit, generate a brief summary:

```
[general] You: /exit

📝 Session Summary:
  • Discussed: Python learning resources, project timeline
  • Goals updated: "Learn Python" - added sub-task "complete tutorial"
  • Learned: You prefer detailed code examples
  • Memory: 4 conversation chunks saved

See you next time!
```

## Implementation

1. Track session activity:
```python
class SessionTracker:
    def __init__(self):
        self.topics_discussed = []
        self.goals_updated = []
        self.preferences_learned = []
        self.queries_count = 0
    
    def add_topic(self, query: str):
        # Extract topic from query
        topic = extract_topic(query)
        if topic and topic not in self.topics_discussed:
            self.topics_discussed.append(topic)
    
    def generate_summary(self) -> str:
        lines = ["📝 Session Summary:"]
        
        if self.topics_discussed:
            lines.append(f"  • Discussed: {', '.join(self.topics_discussed[:3])}")
        
        if self.goals_updated:
            for goal in self.goals_updated:
                lines.append(f"  • Goal updated: \"{goal}\"")
        
        if self.preferences_learned:
            lines.append(f"  • Learned: {self.preferences_learned[0]}")
        
        lines.append(f"  • Memory: {self.queries_count} conversation chunks saved")
        
        return '\n'.join(lines)
```

2. Call on exit:
```python
def handle_exit():
    # Save conversation
    memory.save_conversation(current_messages)
    
    # Generate and show summary
    summary = session_tracker.generate_summary()
    print(f"\n{summary}\n")
    print("See you next time! 👋")
```

## Acceptance Criteria
- [ ] Exit shows what was discussed
- [ ] Exit shows any goals created/updated
- [ ] Exit shows any preferences learned
- [ ] Summary is brief (5 lines max)
```

---

### Prompt 2C: Context-Aware Greetings

```markdown
# Task: Personalized Session Greetings

## Problem  
Senter starts with a generic banner. A phenomenal assistant would greet based on context:
- Time of day
- Last session topics
- Upcoming deadlines
- Learned preferences

## Solution

```
# Morning, returning user with urgent goal:
Good morning! Last time we worked on your investor deck. 
Your "Finish deck by Friday" goal is due in 2 days - want to continue?

# Evening, new topic:
Good evening! What would you like to work on?

# After long absence:
Welcome back! It's been 5 days. Here's what I remember about your projects...
```

## Implementation

```python
def generate_greeting() -> str:
    hour = datetime.now().hour
    time_greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    
    # Check last session
    last_session = memory.get_last_session()
    days_since = (datetime.now() - last_session.date).days if last_session else None
    
    # Check urgent goals
    urgent_goals = [g for g in goal_tracker.get_active_goals() if g.is_urgent()]
    
    # Build greeting
    parts = [f"{time_greeting}!"]
    
    if days_since and days_since > 3:
        parts.append(f"It's been {days_since} days since we last talked.")
    
    if last_session and days_since and days_since < 2:
        parts.append(f"Last time we worked on {last_session.main_topic}.")
    
    if urgent_goals:
        goal = urgent_goals[0]
        parts.append(f"Your \"{goal.description}\" goal is due in {goal.days_until_deadline()} days.")
    
    return ' '.join(parts)
```

## Acceptance Criteria
- [ ] Greeting varies by time of day
- [ ] References last session if recent
- [ ] Mentions urgent goals
- [ ] Feels natural, not robotic
```

---

## PROMPT SET 3: ROBUSTNESS & ERROR HANDLING

### Prompt 3A: Graceful Degradation

```markdown
# Task: Handle Failures Gracefully

## Problem
If Ollama is down, embeddings fail, or web search errors - Senter might crash or give confusing errors.

## Solution
Each feature should fail gracefully and continue working:

```
# If embeddings fail:
[⚠️ Routing unavailable - using default focus]

# If web search fails:
[⚠️ Web search unavailable - answering from knowledge]

# If memory fails:
[⚠️ Memory unavailable - conversation won't be saved]

# If Ollama is down:
❌ Cannot connect to Ollama. Please ensure it's running:
   ollama serve
```

## Implementation

1. Create fallback wrappers:
```python
# Functions/safe_ops.py

def safe_route(query: str, default: str = "general") -> str:
    """Route query with fallback to default"""
    try:
        return embedding_router.route_query(query)
    except Exception as e:
        logging.warning(f"Routing failed: {e}")
        print(f"  [⚠️ Routing unavailable - using {default}]")
        return default

def safe_search(query: str) -> list:
    """Web search with empty fallback"""
    try:
        return web_search.search_web(query)
    except Exception as e:
        logging.warning(f"Web search failed: {e}")
        print("  [⚠️ Web search unavailable]")
        return []

def safe_memory_search(query: str) -> list:
    """Memory search with empty fallback"""
    try:
        return memory.search_memory(query)
    except Exception as e:
        logging.warning(f"Memory search failed: {e}")
        return []

def check_ollama() -> bool:
    """Verify Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False
```

2. Pre-flight check on startup:
```python
def preflight_checks():
    """Run checks before starting"""
    issues = []
    
    if not check_ollama():
        print("❌ Ollama is not running!")
        print("   Start it with: ollama serve")
        sys.exit(1)
    
    if not check_model_available("llama3.2"):
        print("❌ llama3.2 model not found!")
        print("   Install with: ollama pull llama3.2")
        sys.exit(1)
    
    # Optional checks (warn but continue)
    if not check_model_available("nomic-embed-text"):
        print("⚠️ Embedding model not found - routing will use fallback")
    
    print("✓ Pre-flight checks passed")
```

## Acceptance Criteria
- [ ] Ollama down = clear error with fix instructions
- [ ] Embedding fail = uses default focus, continues working
- [ ] Web search fail = continues without web results
- [ ] Memory fail = warns but continues (won't save)
- [ ] Never crashes on recoverable errors
```

---

### Prompt 3B: Input Validation & Edge Cases

```markdown
# Task: Handle Edge Cases and Bad Input

## Problem
- Empty queries
- Very long queries
- Special characters
- Rapid repeated queries
- Malformed commands

## Solution
Validate and handle gracefully:

```python
def validate_query(query: str) -> tuple[bool, str]:
    """Validate user input, return (is_valid, error_message)"""
    
    # Empty
    if not query or not query.strip():
        return False, "Please enter a question or command."
    
    # Too long (token limit concerns)
    if len(query) > 10000:
        return False, "Query too long. Please keep under 10,000 characters."
    
    # Just whitespace/punctuation
    if not any(c.isalnum() for c in query):
        return False, "Please enter a valid question."
    
    return True, ""

def sanitize_query(query: str) -> str:
    """Clean query for processing"""
    # Strip excessive whitespace
    query = ' '.join(query.split())
    
    # Remove null bytes and control characters
    query = ''.join(c for c in query if c.isprintable() or c in '\n\t')
    
    return query

# Rate limiting for API calls
class RateLimiter:
    def __init__(self, max_per_minute: int = 20):
        self.max_per_minute = max_per_minute
        self.calls = []
    
    def can_call(self) -> bool:
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.max_per_minute:
            return False
        self.calls.append(now)
        return True
```

## Acceptance Criteria
- [ ] Empty input gets helpful message
- [ ] Very long input is rejected with explanation
- [ ] Special characters don't crash anything
- [ ] Rate limiting prevents API abuse
- [ ] All edge cases have tests
```

---

## PROMPT SET 4: CONFIGURATION & SETUP

### Prompt 4A: First-Run Setup Wizard

```markdown
# Task: Create Interactive First-Run Setup

## Problem
New users have to manually configure config files. No guidance on what's needed.

## Solution
First-run wizard that configures essentials:

```
🚀 Welcome to Senter! Let's get you set up.

Step 1/3: LLM Backend
  [1] Ollama (local, recommended)
  [2] OpenAI API
  [3] Custom OpenAI-compatible endpoint

Your choice: 1

Checking Ollama... ✓ Found llama3.2

Step 2/3: Your Name
  What should I call you? Jordan

Step 3/3: Primary Use Case
  [1] General assistant
  [2] Coding help
  [3] Research & writing
  [4] All of the above

Your choice: 4

✓ Configuration saved to config/user_profile.json

Ready! Start Senter with: python3 scripts/senter.py
```

## Implementation

```python
# scripts/setup_wizard.py

def run_setup_wizard():
    print("🚀 Welcome to Senter! Let's get you set up.\n")
    
    # Step 1: LLM Backend
    print("Step 1/3: LLM Backend")
    print("  [1] Ollama (local, recommended)")
    print("  [2] OpenAI API")
    print("  [3] Custom endpoint")
    
    choice = input("\nYour choice: ").strip()
    
    if choice == "1":
        config = setup_ollama()
    elif choice == "2":
        config = setup_openai()
    else:
        config = setup_custom()
    
    # Step 2: User name
    print("\nStep 2/3: Your Name")
    name = input("  What should I call you? ").strip() or "User"
    config["user_name"] = name
    
    # Step 3: Use case (sets default focus)
    print("\nStep 3/3: Primary Use Case")
    print("  [1] General assistant")
    print("  [2] Coding help")
    print("  [3] Research & writing")
    print("  [4] All of the above")
    
    use_case = input("\nYour choice: ").strip()
    config["primary_focus"] = {
        "1": "general", "2": "coding", 
        "3": "research", "4": "general"
    }.get(use_case, "general")
    
    # Save
    save_config(config)
    print("\n✓ Configuration saved!")
    print("\nReady! Start Senter with: python3 scripts/senter.py")


def setup_ollama() -> dict:
    """Configure Ollama backend"""
    print("\nChecking Ollama...", end=" ")
    
    if not check_ollama():
        print("❌ Not running")
        print("\nPlease start Ollama first:")
        print("  ollama serve")
        sys.exit(1)
    
    print("✓ Found")
    
    # Check for models
    models = get_ollama_models()
    if "llama3.2" not in models:
        print("\nInstalling llama3.2...")
        os.system("ollama pull llama3.2")
    
    return {
        "central_model": {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434"
        }
    }
```

## Acceptance Criteria
- [ ] First run detects missing config and launches wizard
- [ ] Wizard validates each step before proceeding
- [ ] Ollama setup checks/installs required models
- [ ] OpenAI setup validates API key
- [ ] Generated config is immediately usable
```

---

### Prompt 4B: Configuration Validation

```markdown
# Task: Validate Configuration on Startup

## Problem
Invalid config leads to cryptic errors. Users don't know what's wrong.

## Solution
Validate config and give specific fix instructions:

```python
def validate_config() -> list[str]:
    """Validate config, return list of errors"""
    errors = []
    
    config_path = Path("config/user_profile.json")
    
    if not config_path.exists():
        return ["Config file not found. Run: python3 scripts/setup_wizard.py"]
    
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in config: {e}"]
    
    # Check required fields
    if "central_model" not in config:
        errors.append("Missing 'central_model' in config")
    else:
        model = config["central_model"]
        if "provider" not in model:
            errors.append("Missing 'provider' in central_model")
        
        provider = model.get("provider")
        
        if provider == "ollama":
            if not check_ollama():
                errors.append("Ollama not running. Start with: ollama serve")
            elif not check_model_available(model.get("model", "")):
                errors.append(f"Model '{model.get('model')}' not found. Install with: ollama pull {model.get('model')}")
        
        elif provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                errors.append("OPENAI_API_KEY not set. Export it or add to config.")
    
    return errors

# On startup
errors = validate_config()
if errors:
    print("❌ Configuration errors:\n")
    for error in errors:
        print(f"  • {error}")
    print("\nFix these issues and try again.")
    sys.exit(1)
```

## Acceptance Criteria
- [ ] Missing config → run setup wizard message
- [ ] Invalid JSON → specific error location
- [ ] Missing fields → which field and where
- [ ] Ollama issues → exact fix command
- [ ] OpenAI issues → how to set API key
```

---

## PROMPT SET 5: TUI & POLISH

### Prompt 5A: Get TUI Working

```markdown
# Task: Verify and Fix TUI (senter_app.py)

## Context
The TUI (Textual-based interface) exists but is marked as "untested."
Need to verify it works with all the new features.

## Steps

1. Test current TUI:
```bash
python3 scripts/senter_app.py
```

2. Identify what's broken/missing

3. Integrate new features into TUI:
   - Semantic routing indicator
   - Memory search results panel
   - Goals sidebar
   - Learning preferences display

4. Add keyboard shortcuts:
   - Ctrl+L: Clear screen
   - Ctrl+G: Show goals
   - Ctrl+M: Show memory stats
   - Ctrl+F: Switch focus
   - Ctrl+Q: Quit

## TUI Layout Vision

```
┌─────────────────────────────────────────────────────────┐
│ SENTER v2.0                           [coding] 🧠 📚   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  You: How do I write a Python decorator?                │
│                                                         │
│  [🧠 Memory: 2 relevant conversations]                  │
│  [🎯 Goal: "Learn Python" active]                       │
│                                                         │
│  Senter: A decorator in Python is a function that...    │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ > Type your message...                                  │
├─────────────────────────────────────────────────────────┤
│ ^G Goals  ^M Memory  ^F Focus  ^L Clear  ^Q Quit       │
└─────────────────────────────────────────────────────────┘
```

## Acceptance Criteria
- [ ] TUI starts without errors
- [ ] Can type queries and get responses
- [ ] Focus indicator shows current focus
- [ ] Feature indicators show when active
- [ ] Keyboard shortcuts work
- [ ] Graceful exit with Ctrl+Q
```

---

### Prompt 5B: Add Help System

```markdown
# Task: Comprehensive Help System

## Problem
Users don't know all available commands and features.

## Solution
Add `/help` command with categorized help:

```
[general] You: /help

📖 SENTER HELP

COMMANDS:
  /list              Show available focuses
  /focus <name>      Switch to a focus
  /exit              Save and exit

FEATURES:
  /route <query>     Test semantic routing
  /autoroute         Toggle auto-routing (currently: ON)
  /goals             List active goals
  /memory            Show memory statistics
  /recall <query>    Search past conversations
  /learn             Show learned preferences

TIPS:
  • Mention deadlines to create trackable goals
  • Ask "what do you remember about X?" to test memory
  • Say "I prefer..." to teach preferences

Type /help <command> for detailed help on a specific command.
```

## Implementation

```python
HELP_TOPICS = {
    "main": """
📖 SENTER HELP

COMMANDS:
  /list              Show available focuses
  /focus <name>      Switch to a focus
  /exit              Save and exit

FEATURES:
  /route <query>     Test semantic routing
  /autoroute         Toggle auto-routing
  /goals             List active goals  
  /memory            Show memory statistics
  /recall <query>    Search past conversations
  /learn             Show learned preferences

Type /help <topic> for more details.
""",
    
    "goals": """
🎯 GOAL TRACKING

Senter automatically extracts goals from your conversations.

Examples that create goals:
  • "I need to finish my report by Friday"
  • "I want to learn Python this month"
  • "Remind me to call John tomorrow"

Commands:
  /goals           List all active goals
  /goals done <n>  Mark goal #n as complete
  /goals delete <n> Delete goal #n

Goals are saved between sessions and surface when relevant.
""",
    
    "memory": """
🧠 CONVERSATION MEMORY

Senter remembers your past conversations.

Commands:
  /memory          Show statistics
  /recall <query>  Search past conversations
  /forget          Clear all memory (requires confirmation)

Memory is used automatically to provide context in responses.
Ask "what do you remember about X?" to test recall.
""",
}

def handle_help(args: str):
    topic = args.strip().lower() if args else "main"
    if topic in HELP_TOPICS:
        print(HELP_TOPICS[topic])
    else:
        print(f"No help available for '{topic}'")
        print("Available topics: " + ", ".join(HELP_TOPICS.keys()))
```

## Acceptance Criteria
- [ ] /help shows main help
- [ ] /help <topic> shows detailed help
- [ ] All commands documented
- [ ] Tips for effective usage included
```

---

## EXECUTION PRIORITY

### Quick Wins (1-2 hours each)
1. **Prompt 1A**: Silence warnings — immediate UX improvement
2. **Prompt 3A**: Graceful degradation — prevents crashes
3. **Prompt 5B**: Help system — users can self-serve

### Medium Effort (2-4 hours each)
4. **Prompt 1B**: Feature indicators — shows value of features
5. **Prompt 2B**: Session summaries — satisfying exit experience
6. **Prompt 3B**: Input validation — robustness

### Larger Effort (4-6 hours each)
7. **Prompt 2A**: Proactive goals — differentiating feature
8. **Prompt 2C**: Smart greetings — feels intelligent
9. **Prompt 4A**: Setup wizard — smooth onboarding
10. **Prompt 5A**: TUI working — visual interface

---

## THE PERFECT SENTER SESSION (After Polish)

```
$ python3 scripts/senter.py

Good afternoon, Jordan! Last time we worked on your investor deck.
⚠️ Your "Finish deck by Friday" goal is due in 2 days.

📋 Active goals (2):
  ⚠️ "Finish deck by Friday" (2 days left)
  • "Learn Python by February" (24 days left)

[general] You: Let's work on the deck. What sections are left?

  [🧠 Memory: Found 3 relevant conversations]
  [🎯 Goal: "Finish deck by Friday" active]

Senter: Based on our last session, you've completed the problem and solution 
slides. Still remaining:
  • Market size (TAM/SAM/SOM) - you mentioned needing research
  • Team slide - waiting on headshots from Sarah
  • Financial projections - draft exists but needs review

Want me to research current AI market sizing for the TAM slide?

[general] You: Yes, research that and give me bullet points

  [🔍 Research: Searching in parallel...]

Senter: Here's what I found for your TAM slide:

**Global AI Market Size:**
• 2024: $184B → 2030: $826B (CAGR 28.5%) [web: statista.com]
• Enterprise AI assistants segment: $12B → $47B [web: gartner.com]

**Relevant for your pitch:**
• "AI productivity tools" specifically: $5.4B market in 2024
• Growing 34% annually - faster than general AI market

I've saved this research to your deck focus for reference.

[general] You: Perfect. /exit

📝 Session Summary:
  • Discussed: Investor deck progress, market research
  • Research saved: AI market sizing data
  • Goal status: "Finish deck" - 2 sections remaining

See you next time! 👋
```

**That's polish.** The features work, the feedback is clear, and it feels like a capable assistant.

---

**End of Document**
