# Senter: Autonomous Research - Implementation Plan

## The Goal

**One sentence:** When you mention something interesting in conversation, Senter researches it while you're away and has a summary waiting when you return.

**The experience:**
```
Monday 10pm: You chat with Senter about a work problem. You mention "I wonder if event sourcing would help here."

Tuesday 7am: You open your laptop. Menubar icon pulses blue.
             Click → "I researched event sourcing patterns. Found 4 relevant articles. Want the summary?"
             Click "Show me" → Clean 2-minute read with sources.
             Click "Useful" → Senter learns you care about architecture patterns.
```

---

## Current State vs. Target State

| Aspect | Current State | Target State |
|--------|---------------|--------------|
| Topic Detection | Keyword grep ("research", "learn about") | Semantic extraction using LLM |
| Research Action | Ask Ollama a question | Web search + read articles + synthesize |
| Output Quality | Raw LLM response | Structured summary with citations |
| Presentation | CLI command to view | Menubar notification when ready |
| Feedback | None | Thumbs up/down that improves future |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENTER DAEMON                             │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│  │ Conversation  │───▶│ Topic Extractor  │───▶│ Research     │ │
│  │ Logger        │    │ (LLM-based)      │    │ Queue        │ │
│  └───────────────┘    └──────────────────┘    └──────┬───────┘ │
│                                                       │         │
│                       ┌───────────────────────────────▼───────┐ │
│                       │         RESEARCH WORKER               │ │
│                       │  ┌─────────┐  ┌──────────┐  ┌───────┐│ │
│                       │  │ Web     │  │ Content  │  │Synth- ││ │
│                       │  │ Search  │  │ Fetcher  │  │esizer ││ │
│                       │  └─────────┘  └──────────┘  └───────┘│ │
│                       └───────────────────────────────────────┘ │
│                                          │                      │
│  ┌──────────────────────────────────────▼────────────────────┐ │
│  │                    RESEARCH STORE                          │ │
│  │  { topic, sources[], summary, key_points[], created_at }  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                          │                      │
│                                          │ IPC Socket           │
└──────────────────────────────────────────┼──────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │   MENUBAR APP (UI)     │
                              │  ┌──────────────────┐  │
                              │  │ 🔵 Senter        │  │
                              │  │ ─────────────────│  │
                              │  │ New: Event Sourc │  │
                              │  │ ─────────────────│  │
                              │  │ Previous (3)     │  │
                              │  │ Settings         │  │
                              │  │ Quit             │  │
                              │  └──────────────────┘  │
                              └────────────────────────┘
```

---

## Implementation Stories

### Phase 1: Research Engine (The Brain)

#### RE-001: Semantic Topic Extraction
**Current:** Regex for "research X", "learn about Y"
**Target:** LLM analyzes conversation and extracts research-worthy topics

```python
# Input: Last 10 conversation turns
# Output: [{"topic": "event sourcing", "reason": "user expressed curiosity", "priority": 0.8}]
```

**Acceptance Criteria:**
- Extracts topics user shows genuine curiosity about (not just mentions)
- Assigns priority based on signals (questions, "I wonder", repeated mentions)
- Deduplicates with existing research queue
- Works with Ollama locally

#### RE-002: Deep Web Research
**Current:** Single DuckDuckGo search, return raw results
**Target:** Multi-query search, fetch actual content, extract key info

```python
# Input: topic = "event sourcing patterns"
# Process:
#   1. Generate 3-5 search queries for different angles
#   2. Search each, collect top 10 unique URLs
#   3. Fetch page content (readability extraction)
#   4. Extract key points from each source
# Output: [{url, title, key_points[], relevance_score}]
```

**Acceptance Criteria:**
- Generates diverse search queries (intro, examples, comparisons, gotchas)
- Fetches actual page content, not just snippets
- Handles failures gracefully (skip failed URLs)
- Respects rate limits and timeouts
- Stores raw sources for citation

#### RE-003: Multi-Source Synthesis
**Current:** ResearchResultSummarizer does extractive summary of single text
**Target:** LLM synthesizes multiple sources into coherent summary

```python
# Input: 5-10 source documents with key points
# Output: {
#   "summary": "2-3 paragraph synthesis",
#   "key_points": ["Point 1 [source1, source2]", "Point 2 [source3]"],
#   "practical_takeaways": ["Do X", "Avoid Y"],
#   "sources": [{url, title, relevance}]
# }
```

**Acceptance Criteria:**
- Synthesizes, doesn't just concatenate
- Cites sources for each key point
- Identifies consensus vs. disagreement
- Produces actionable takeaways
- Summary is 2-3 minute read max

#### RE-004: Research Quality Scoring
**Current:** None
**Target:** Score research quality before presenting

```python
# Factors:
#   - Source diversity (not all from same domain)
#   - Recency (prefer recent for tech topics)
#   - Depth (enough substance to be useful)
#   - Relevance (matches original topic intent)
# Output: quality_score 0-1, with breakdown
```

**Acceptance Criteria:**
- Low-quality research is retried or discarded
- Quality threshold configurable
- Breakdown stored for debugging

---

### Phase 2: Menubar UI (The Interface)

#### UI-001: Menubar App Foundation
**Stack:** Python + rumps (lightweight macOS menubar framework)

**Layout:**
```
🔵 Senter
├── ✨ New Research (1)     → Shows latest
├── 📚 Previous Research    → List of past 10
├── ⚙️  Settings            → Opens config
├── ───────────────────
└── Quit
```

**Acceptance Criteria:**
- App starts with daemon, lives in menubar
- Icon changes color when new research ready (blue pulse)
- Clicking opens dropdown, not full window
- Communicates with daemon via IPC socket

#### UI-002: Research Summary View
**Trigger:** Click on a research item
**Display:** Native macOS panel (not Electron, not browser)

```
┌─────────────────────────────────────────────────────┐
│  Event Sourcing Patterns                       [×]  │
├─────────────────────────────────────────────────────┤
│  Researched: 2 hours ago | 4 sources | 3 min read  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Event sourcing stores state changes as a sequence │
│  of events rather than current state...            │
│                                                     │
│  Key Points:                                        │
│  • Events are immutable and append-only [1,2]      │
│  • Enables complete audit trail [1,3]              │
│  • Complexity increases for queries [2,4]          │
│                                                     │
│  Practical Takeaways:                              │
│  ✓ Good for: audit requirements, complex domains   │
│  ✗ Avoid if: simple CRUD, tight deadlines          │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Sources:                                          │
│  [1] martinfowler.com/eaaDev/EventSourcing         │
│  [2] eventstore.com/blog/what-is-event-sourcing    │
│  [3] microservices.io/patterns/data/event-sourcing │
│  [4] stackoverflow.com/questions/12345             │
├─────────────────────────────────────────────────────┤
│  [👍 Useful]  [👎 Not helpful]  [🔄 Research more] │
└─────────────────────────────────────────────────────┘
```

**Acceptance Criteria:**
- Clean, readable typography
- Sources are clickable links
- Feedback buttons work
- Panel dismissible with Escape or click outside

#### UI-003: Feedback Integration
**When user clicks feedback:**
- 👍 Useful → Record topic as high-value, boost similar future research
- 👎 Not helpful → Record why (too shallow? wrong angle?), learn to avoid
- 🔄 Research more → Add to queue for deeper dive

**Acceptance Criteria:**
- Feedback stored in learning database
- Affects future topic prioritization
- Optional: Ask "what was missing?" on negative feedback

---

### Phase 3: Integration & Polish

#### INT-001: Daemon Integration
- Research worker runs in daemon's process pool
- Triggered by scheduler (configurable interval, default: check every 30 min)
- Also triggered when attention lost (user looks away = start researching)

#### INT-002: Conversation Logging for Topics
- All conversations logged to EventsDB (already exists)
- Topic extractor runs on recent conversation history
- Deduplication against research already done

#### INT-003: Glassmorphism Styling (from old Senter)
- If native panels look dated, use PyQt5/6 with custom styling
- Translucent backgrounds, subtle borders, blur effects
- Match the aesthetic of the old Next.js app

---

## File Changes

### New Files
```
ui/
├── __init__.py
├── menubar_app.py      # Main menubar application (rumps)
├── research_panel.py   # Research summary panel (PyQt or native)
└── styles.py           # Glassmorphism CSS/styling

research/
├── __init__.py
├── topic_extractor.py   # Semantic topic extraction (LLM)
├── deep_researcher.py   # Multi-query web research
├── synthesizer.py       # Multi-source synthesis (LLM)
├── quality_scorer.py    # Research quality evaluation
└── research_store.py    # SQLite storage for research results
```

### Modified Files
```
daemon/senter_daemon.py   # Add UI process, integrate research worker
daemon/ipc_server.py      # Add endpoints for UI communication
scheduler/research_trigger.py  # Wire up new research engine
config/daemon_config.json # Add research/UI settings
```

---

## Tech Stack Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Menubar | rumps | Lightweight, native macOS, Python |
| Research Panel | PyQt6 + custom CSS | Glassmorphism possible, stays in Python |
| Web Fetching | httpx + readability-lxml | Async + content extraction |
| LLM | Ollama (existing) | Already integrated |
| Storage | SQLite | Already used for events, simple |

---

## Success Metrics

1. **Topic Detection Accuracy:** >80% of extracted topics are genuinely interesting
2. **Research Quality:** >70% of research summaries rated "useful" by user
3. **Latency:** Research completes within 5 minutes of topic detection
4. **Reliability:** <5% of research attempts fail completely

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Web scraping blocked | Multiple user-agents, respect robots.txt, use cached when available |
| LLM too slow | Queue and batch, show "researching..." status |
| Topic extraction too noisy | Start conservative, require strong signals |
| UI feels disconnected | Tight IPC, real-time status updates |

---

## Implementation Order

1. **RE-001** Topic Extraction → Get quality topics first
2. **RE-002** Deep Research → Actually gather information
3. **RE-003** Synthesis → Make it readable
4. **UI-001** Menubar → Basic notification
5. **UI-002** Panel → Show the research
6. **RE-004** Quality Scoring → Filter bad research
7. **UI-003** Feedback → Close the loop
8. **INT-001-003** Polish → Make it seamless

---

## Questions to Resolve

1. **Frequency:** How often should topic extraction run? (Every conversation? Every 5 min?)
2. **Depth:** How many sources is enough? (3? 10? 20?)
3. **Notification:** When exactly to notify? (Immediately? Batch daily?)
4. **History:** How long to keep research? (Forever? 30 days?)
