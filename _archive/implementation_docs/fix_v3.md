# SENTER: From Demo to Phenomenal

**Purpose**: Exact prompts for Claude Code + vision of what "fully functional" means  
**Current State**: Demo-ready (CLI works, Ollama responds, 8/8 tests pass)  
**Gap**: ~30% functional, ~70% stubs/prompts/aspirational

---

## PART 1: WHAT "PHENOMENALLY WORKING" LOOKS LIKE

### The Vision vs. Current Reality

| Aspiration | Current Reality | Phenomenal State |
|------------|-----------------|------------------|
| "7 working agents" | 7 SENTER.md prompt templates | 7 Python classes with actual logic that coordinate |
| "Self-learning" | `print("evolution check completed")` | Analyzes conversations, extracts patterns, updates configs |
| "Semantic routing" | Returns first N items | Embeds query, compares to focus embeddings, routes intelligently |
| "Parallel processing" | Background threads exist | Dual inference: one answers, one researches simultaneously |
| "Always working for you" | Responds when asked | Background task queue, proactive suggestions, goal tracking |
| "Learns your preferences" | Static config file | Dynamic profile that updates from every interaction |

### A Phenomenally Working Senter Session

```
$ python3 scripts/senter.py

[Senter] Good morning, Jordan. Based on our conversations:
  • You have 3 active goals I'm tracking
  • Yesterday you mentioned wanting to research AI agents - I found 4 relevant papers overnight
  • Your "coding" focus has learned you prefer TypeScript examples

> What's the status on my investor deck project?

[Routing → Goal_Detector → Context_Gatherer → Planner]

[Senter] Here's what I know about your investor deck:
  • Goal created: Dec 15, 2025
  • Last discussed: Jan 5, 2026 (you asked about market size slides)
  • Related context: You mentioned Series A targeting, $2M raise
  • Outstanding tasks:
    1. Finalize TAM/SAM/SOM numbers (you said "need to research")
    2. Add team slide (noted as "waiting on headshots")
  
  Want me to research current AI market sizing data for your TAM slide?

> Yes, and also draft the competitive landscape section

[Senter] Starting parallel tasks:
  • [Research Worker] Searching for AI market size 2025-2026...
  • [Draft Worker] Loading your deck context and competitor notes...

[2.3 seconds later]

[Senter] Here's what I found and drafted:

**Market Research** (3 sources):
  • Global AI market: $196B in 2025, projected $739B by 2030 (Statista)
  • Enterprise AI assistants specifically: $12.4B → $47B (Gartner)
  • [Saved to: Focuses/research/investor_deck_market_data.md]

**Competitive Landscape Draft**:
  [Shows draft based on your notes + fresh research]
  
  This draft incorporates the 4 competitors you mentioned previously 
  (ChatGPT, Notion AI, Jasper, Copy.ai) plus 2 new relevant ones I found.

[Learning] Updated your profile:
  • Added "investor deck" to active projects
  • Noted preference for data-backed claims
  • Linked market research to your Series A goal
```

**That's phenomenal.** It:
- Remembers context across sessions
- Tracks goals proactively  
- Routes queries to appropriate handlers
- Runs parallel tasks
- Learns from interactions
- Provides actionable output

---

## PART 2: THE GAP ANALYSIS

### What Exists vs. What's Needed

| Component | Exists | Needed for Phenomenal |
|-----------|--------|----------------------|
| CLI interface | ✅ Working | Minor UX improvements |
| LLM integration | ✅ Ollama works | Multi-model support (research + response) |
| Focus system | ✅ 5 focuses load | Focus-specific memory + embeddings |
| SENTER.md parser | ✅ Works | Dynamic updating from learning |
| Web search | ⚠️ Limited API | Better search (Brave/Tavily) or duckduckgo-search package |
| Router | ❌ Prompt only | Embedding-based Python routing |
| Goal tracking | ❌ Doesn't exist | Goal extraction + persistence + tracking |
| Self-learning | ❌ Stub | Conversation analysis → config updates |
| Context memory | ❌ Doesn't exist | Vector store of past conversations |
| Parallel inference | ❌ Threads exist, not used | Actual dual-model queries |
| User profile | ❌ Static JSON | Dynamic profile that grows |

### Effort Estimates

| Feature | Complexity | Time Estimate |
|---------|------------|---------------|
| Real semantic routing | Medium | 4-6 hours |
| Goal extraction + tracking | Medium | 6-8 hours |
| Conversation memory (vector store) | Medium | 4-6 hours |
| Self-learning (basic) | High | 8-12 hours |
| Parallel inference | Medium | 4-6 hours |
| Better web search | Low | 2-3 hours |
| Dynamic user profile | Medium | 4-6 hours |

**Total to "Phenomenal"**: ~40-50 hours of focused development

---

## PART 3: EXACT PROMPTS FOR CLAUDE CODE

### Prompt 1: Implement Real Semantic Routing

```markdown
# Task: Implement Embedding-Based Semantic Routing

## Context
Currently, Senter's routing is either keyword-based or just sends everything to one handler.
The `_embed_filter()` function in `scripts/senter_selector.py` is a stub that returns the first N items.

## Requirements
1. Use Ollama's embedding model (nomic-embed-text or mxbai-embed-large)
2. On startup, generate embeddings for each Focus based on its SENTER.md content
3. When a query comes in:
   - Embed the query
   - Compare to all focus embeddings (cosine similarity)
   - Route to the best matching focus
4. Cache focus embeddings (regenerate only when SENTER.md changes)

## Implementation Steps
1. Create `Functions/embedding_router.py` with:
   - `embed_text(text: str) -> list[float]` - calls Ollama embedding API
   - `load_or_generate_focus_embeddings() -> dict[str, list[float]]`
   - `route_query(query: str) -> str` - returns best focus name
   
2. Update `scripts/senter.py` to use the new router before processing queries

3. Test with queries that should route to different focuses:
   - "Help me write a poem" → creative
   - "Debug this Python code" → coding  
   - "What's the latest news on AI" → research

## Acceptance Criteria
- [ ] Queries route to appropriate focuses based on semantic similarity
- [ ] Routing decision is logged/visible
- [ ] Focus embeddings are cached (not regenerated every query)
- [ ] Falls back gracefully if embedding fails

## Files to Modify/Create
- CREATE: Functions/embedding_router.py
- MODIFY: scripts/senter.py (integrate router)
- MODIFY: scripts/senter_selector.py (replace stub)
```

---

### Prompt 2: Implement Goal Tracking System

```markdown
# Task: Implement Goal Extraction and Tracking

## Context
Senter should track user goals across sessions. Currently there's no goal persistence.
The Goal_Detector "agent" is just a SENTER.md prompt template with no backing code.

## Requirements
1. Extract goals from conversations using the LLM
2. Persist goals to a JSON file (goals.json)
3. Track goal status: active, completed, abandoned
4. Surface relevant goals when context matches

## Data Structure
```json
{
  "goals": [
    {
      "id": "goal_001",
      "description": "Complete investor deck for Series A",
      "created": "2025-12-15T10:30:00Z",
      "status": "active",
      "related_conversations": ["conv_123", "conv_456"],
      "sub_tasks": [
        {"task": "Research TAM numbers", "status": "pending"},
        {"task": "Get team headshots", "status": "waiting"}
      ],
      "last_mentioned": "2026-01-05T14:22:00Z"
    }
  ]
}
```

## Implementation Steps
1. Create `Functions/goal_tracker.py` with:
   - `extract_goals_from_message(message: str, response: str) -> list[Goal]`
   - `save_goal(goal: Goal)`
   - `get_active_goals() -> list[Goal]`
   - `get_relevant_goals(query: str) -> list[Goal]` (uses embeddings)
   - `update_goal_status(goal_id: str, status: str)`

2. Create `data/goals.json` for persistence

3. Integrate into main loop:
   - After each conversation turn, check for new goals
   - Before responding, check for relevant existing goals
   - Include goal context in system prompt when relevant

## Acceptance Criteria
- [ ] Goals are extracted from conversations that mention intentions/plans
- [ ] Goals persist across sessions
- [ ] Relevant goals surface when discussing related topics
- [ ] User can ask "what are my current goals?" and get accurate list

## Files to Create/Modify
- CREATE: Functions/goal_tracker.py
- CREATE: data/goals.json
- MODIFY: scripts/senter.py (integrate goal tracking)
- MODIFY: Focuses/internal/Goal_Detector/SENTER.md (update prompt)
```

---

### Prompt 3: Implement Conversation Memory

```markdown
# Task: Implement Persistent Conversation Memory with Vector Search

## Context
Senter should remember past conversations and retrieve relevant context.
Currently there's no conversation persistence or retrieval.

## Requirements
1. Save all conversations with timestamps
2. Generate embeddings for conversation chunks
3. Retrieve relevant past context when processing new queries
4. Respect a context window limit (don't overload the LLM)

## Implementation Steps
1. Create `Functions/memory.py` with:
   - `save_conversation(messages: list[dict])` - saves to JSON + generates embeddings
   - `search_memory(query: str, limit: int = 5) -> list[ConversationChunk]`
   - `get_recent_conversations(n: int = 10) -> list[Conversation]`
   - `summarize_conversation(conversation: Conversation) -> str` (for long convos)

2. Create `data/conversations/` directory structure:
   ```
   data/conversations/
   ├── index.json           # Metadata for all conversations
   ├── embeddings.npy       # Numpy array of all chunk embeddings  
   ├── 2026-01-07_001.json  # Individual conversation files
   └── 2026-01-07_002.json
   ```

3. Integrate into main loop:
   - Before responding, search memory for relevant context
   - Include relevant context in system prompt (with source attribution)
   - After conversation ends, save and embed it

## Acceptance Criteria
- [ ] Conversations are saved after each session
- [ ] Relevant past context surfaces when discussing similar topics
- [ ] "What did we discuss about X?" returns accurate results
- [ ] Memory search is fast (<500ms)

## Files to Create/Modify
- CREATE: Functions/memory.py
- CREATE: data/conversations/index.json
- MODIFY: scripts/senter.py (save conversations, inject context)
```

---

### Prompt 4: Implement Self-Learning

```markdown
# Task: Implement Basic Self-Learning System

## Context
The `_evolve_agents()` function is a stub that just prints a log message.
Senter should learn from conversations and update its behavior.

## What "Learning" Means (Concrete)
1. **Preference Learning**: Notice patterns in user requests/feedback
2. **Style Learning**: Adapt response length, formality, detail level
3. **Topic Learning**: Track what topics user engages with most
4. **Workflow Learning**: Remember how user likes tasks done

## Implementation Steps
1. Create `Functions/learner.py` with:
   - `analyze_conversation(messages: list) -> Insights`
   - `update_user_profile(insights: Insights)`
   - `update_focus_config(focus: str, insights: Insights)` 
   - `get_learned_preferences() -> dict`

2. Define learning triggers:
   - After every N conversations (batch learning)
   - When user explicitly corrects or praises response
   - When user repeatedly asks for modifications

3. What gets updated:
   ```json
   // config/user_profile.json additions
   {
     "learned_preferences": {
       "response_length": "detailed",  // learned from "can you expand on that?"
       "code_language": "TypeScript",  // learned from request patterns
       "formality": "casual",          // learned from user's tone
       "topics_of_interest": ["AI", "startups", "productivity"]
     },
     "interaction_patterns": {
       "average_session_length": 12,
       "peak_usage_hours": [9, 10, 14, 15],
       "common_requests": ["research", "code review", "writing"]
     }
   }
   ```

4. Replace the stub in `scripts/background_processor.py`:
   ```python
   def _evolve_agents(self):
       from Functions.learner import analyze_recent_conversations, apply_learnings
       insights = analyze_recent_conversations(last_n=10)
       apply_learnings(insights)
       self.logger.info(f"Applied {len(insights)} learnings")
   ```

## Acceptance Criteria
- [ ] After 5+ conversations, user profile shows learned preferences
- [ ] Response style adapts based on learned preferences
- [ ] Learning is logged and visible (user can see what was learned)
- [ ] User can correct/override learned preferences

## Files to Create/Modify
- CREATE: Functions/learner.py
- MODIFY: config/user_profile.json (add learned_preferences schema)
- MODIFY: scripts/background_processor.py (replace stub)
- MODIFY: Functions/omniagent.py (apply preferences to responses)
```

---

### Prompt 5: Implement Parallel Inference

```markdown
# Task: Implement True Parallel Dual-Worker Inference

## Context
The background_processor.py has threading for 4 tasks, but doesn't do parallel inference.
The vision is: one worker answers the user, another does background research simultaneously.

## Requirements
1. When a query comes in that would benefit from research:
   - Worker 1: Start generating response immediately
   - Worker 2: Search web/memory for relevant context
   - Merge: Incorporate research into response (or append as addendum)

2. For simple queries, don't add overhead - just respond directly

## Implementation Steps
1. Create `Functions/parallel_inference.py` with:
   - `needs_research(query: str) -> bool` - heuristic for research-worthy queries
   - `parallel_respond(query: str) -> Response` - orchestrates dual workers
   - Uses `concurrent.futures.ThreadPoolExecutor` or `asyncio`

2. Research worker does:
   - Web search (if query is about current events/facts)
   - Memory search (if query might relate to past conversations)
   - Document search (if user has uploaded files)

3. Response worker does:
   - Normal LLM inference
   - Streams response while research happens
   - Can be interrupted to incorporate research findings

4. Merge strategies:
   - **Inline**: Research findings woven into response
   - **Addendum**: Response first, then "I also found..."
   - **Replacement**: If research contradicts initial response, correct it

## Acceptance Criteria
- [ ] Research-worthy queries trigger parallel processing
- [ ] Simple queries don't have added latency
- [ ] Research findings visibly improve responses
- [ ] User can see when parallel processing is happening

## Files to Create/Modify
- CREATE: Functions/parallel_inference.py
- MODIFY: scripts/senter.py (use parallel inference for queries)
- MODIFY: Functions/omniagent.py (support streaming + interruption)
```

---

### Prompt 6: Fix Web Search Properly

```markdown
# Task: Replace Limited DuckDuckGo Instant Answer API with Proper Search

## Context
Current web_search.py uses DuckDuckGo Instant Answer API which returns empty for most queries.
Need a search solution that actually works.

## Options (in order of preference)
1. **duckduckgo-search package** (free, no API key)
2. **Brave Search API** (free tier: 2000/month)
3. **Tavily API** (free tier: 1000/month, great for AI)
4. **SearXNG** (self-hosted, completely free)

## Implementation (using duckduckgo-search)
```python
# Functions/web_search.py - replacement

from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Perform web search using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def search_news(query: str, max_results: int = 5) -> list[dict]:
    """Search recent news"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
            return results
    except Exception as e:
        print(f"News search error: {e}")
        return []
```

## Steps
1. Install: `pip install duckduckgo-search`
2. Replace contents of Functions/web_search.py
3. Test with various queries
4. Update requirements.txt

## Acceptance Criteria
- [ ] "Search for latest AI news" returns actual results
- [ ] "What is the current price of Bitcoin" returns data
- [ ] Search results are incorporated into responses
- [ ] Graceful fallback on rate limiting

## Files to Modify
- MODIFY: Functions/web_search.py (complete replacement)
- MODIFY: requirements.txt (add duckduckgo-search)
```

---

### Prompt 7: Create the Full Integration Test

```markdown
# Task: Create Comprehensive Integration Test Suite

## Context
Current tests only verify components work in isolation.
Need tests that verify the full "phenomenal" experience.

## Test Scenarios

### Test 1: Semantic Routing
```python
def test_semantic_routing():
    queries = [
        ("Write me a haiku about coding", "creative"),
        ("Debug this Python function", "coding"),
        ("What's happening in AI today", "research"),
        ("Tell me about yourself", "general"),
    ]
    for query, expected_focus in queries:
        actual_focus = router.route_query(query)
        assert actual_focus == expected_focus, f"'{query}' routed to {actual_focus}, expected {expected_focus}"
```

### Test 2: Goal Tracking
```python
def test_goal_tracking():
    # Simulate conversation mentioning a goal
    messages = [
        {"role": "user", "content": "I need to finish my presentation by Friday"},
        {"role": "assistant", "content": "I'll help you with your presentation..."}
    ]
    
    # Extract goals
    goals = goal_tracker.extract_goals_from_conversation(messages)
    assert len(goals) >= 1
    assert "presentation" in goals[0].description.lower()
    
    # Verify persistence
    goal_tracker.save_goal(goals[0])
    loaded = goal_tracker.get_active_goals()
    assert any("presentation" in g.description.lower() for g in loaded)
```

### Test 3: Memory Retrieval
```python
def test_memory_retrieval():
    # Save a conversation about a specific topic
    memory.save_conversation([
        {"role": "user", "content": "My favorite programming language is Rust"},
        {"role": "assistant", "content": "Rust is great for systems programming..."}
    ])
    
    # Query should retrieve relevant context
    results = memory.search_memory("What programming languages do I like?")
    assert len(results) > 0
    assert "Rust" in results[0].content
```

### Test 4: End-to-End Flow
```python
def test_end_to_end():
    # Full flow: query → route → context → respond → learn → save
    result = senter.process_query("Help me plan my week")
    
    assert result.response is not None
    assert result.focus_used in ["general", "planning"]
    assert result.context_retrieved >= 0
    assert result.conversation_saved == True
```

## Implementation
Create `tests/test_integration.py` with all scenarios above.
Use pytest fixtures for setup/teardown.

## Acceptance Criteria
- [ ] All 4 test scenarios pass
- [ ] Tests can run in CI (no manual intervention)
- [ ] Tests complete in <30 seconds
- [ ] Clear failure messages when something breaks
```

---

## PART 4: PRIORITY ORDER

### If You Have 8 Hours

```
1. [2h] Prompt 6: Fix Web Search (immediate value, low effort)
2. [4h] Prompt 1: Semantic Routing (makes focuses actually useful)
3. [2h] Prompt 3: Conversation Memory (basic version)
```

**Result**: A Senter that routes queries intelligently, searches the web properly, and remembers past conversations.

### If You Have 20 Hours

Add:
```
4. [6h] Prompt 2: Goal Tracking
5. [4h] Prompt 4: Self-Learning (basic version)
6. [4h] Prompt 5: Parallel Inference
```

**Result**: A Senter that tracks your goals, learns your preferences, and does research in parallel.

### If You Have 40+ Hours

Add:
```
7. [4h] Prompt 7: Full Integration Tests
8. [8h] Polish: Error handling, UX improvements, documentation
9. [8h] TUI: Get senter_app.py fully working with all features
10. [8h] Voice: Integrate TTS/STT properly
```

**Result**: A polished, tested, phenomenal Senter.

---

## PART 5: THE HONEST CONVERSATION

### What to Tell Stakeholders

**Current State (After v1+v2)**:
> "Senter is a working prototype. The CLI connects to a local LLM (Ollama), queries get responses, and we have 5 configurable focus areas. Core infrastructure is solid."

**What "7 Agents" Actually Means**:
> "We have 7 SENTER.md configuration files that define different AI behaviors. They're prompt templates that shape how the LLM responds, not standalone programs. Think of them as 'personalities' the system can adopt."

**What's Real vs. Roadmap**:
> "Real: CLI, LLM integration, focus switching, basic web search. Roadmap: semantic routing, goal tracking, self-learning, parallel inference. We have the architecture for these, but the implementations are stubs."

**Time to "Phenomenal"**:
> "With focused development, 40-50 hours to implement the core differentiating features. That's 1-2 weeks of dedicated work, not months."

---

## SUMMARY

### The Perfect Questions for Claude Code

1. **"Implement embedding-based semantic routing"** (Prompt 1)
2. **"Build goal extraction and tracking"** (Prompt 2)
3. **"Create conversation memory with vector search"** (Prompt 3)
4. **"Implement self-learning from conversations"** (Prompt 4)
5. **"Build parallel inference for research queries"** (Prompt 5)
6. **"Fix web search with duckduckgo-search package"** (Prompt 6)
7. **"Create integration tests for all features"** (Prompt 7)

### What Phenomenal Looks Like

- **Remembers**: Past conversations, goals, preferences
- **Routes**: Queries to the right focus based on meaning
- **Learns**: Adapts style and behavior from interactions
- **Works**: Background research while responding
- **Tracks**: Goals and tasks across sessions
- **Searches**: Web actually works and enhances responses

### The Gap

| From | To | Effort |
|------|-----|--------|
| Demo (now) | Useful Daily Tool | ~20 hours |
| Useful Daily Tool | Phenomenal | ~20 more hours |
| Phenomenal | Production Ready | ~40 more hours |

---

**End of Document**

#### CHAT BACKGROUND CONTEXT:
"<This is the right question. Let me create a comprehensive document that gives you both: the exact prompts for Claude Code AND a vision of what "phenomenally working" actually means for Senter.Here's **senter_to_phenomenal.md** — the complete roadmap from demo to phenomenal.

## What You're Getting

### Part 1: What "Phenomenal" Actually Looks Like
A concrete example session showing Senter that:
- Greets you with tracked goals
- Remembers context from past conversations
- Routes queries to the right handler
- Runs parallel research while responding
- Updates your profile based on interactions

### Part 2: The Gap Analysis
Exactly what exists vs. what's needed:

| Feature | Now | Phenomenal |
|---------|-----|------------|
| Routing | Sends everything to Ollama | Embedding-based focus selection |
| Memory | None | Vector store of conversations |
| Goals | Nothing | Extraction + persistence + tracking |
| Learning | `print("check completed")` | Profile updates from interactions |
| Search | Empty API responses | Actual working search |

### Part 3: The 7 Perfect Prompts

Each one is a complete, copy-paste-ready prompt for Claude Code:

1. **Semantic Routing** — Embed queries, compare to focuses, route intelligently
2. **Goal Tracking** — Extract goals from conversations, persist, surface when relevant
3. **Conversation Memory** — Save, embed, search past conversations
4. **Self-Learning** — Analyze patterns, update preferences, adapt behavior
5. **Parallel Inference** — One worker responds, another researches
6. **Fix Web Search** — Replace broken API with working package
7. **Integration Tests** — Verify the full "phenomenal" experience

### Part 4: Priority Order

**8 hours**: Web search + Routing + Basic memory → *Actually useful*

**20 hours**: Add goals + learning + parallel → *Differentiated product*

**40+ hours**: Polish, tests, TUI, voice → *Production ready*

### Part 5: The Honest Conversation

What to tell stakeholders about current state vs. roadmap.

---

## The Bottom Line

**Current Senter**: A CLI that talks to Ollama with configurable prompt templates.

**Phenomenal Senter**: An AI that remembers you, tracks your goals, learns your preferences, researches while answering, and gets smarter over time.

**The gap**: ~40-50 hours of focused development using the 7 prompts provided.

You now have everything you need to either:
1. Give these prompts to Claude Code and build it yourself
2. Have an honest conversation with Chris about what actually needs to be built
3. Make informed decisions about whether to continue investing in this project>""