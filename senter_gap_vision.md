

# Senter Vision Question: DO NOT OVERWRITE THIS FILE. ONLY ADD TO IT.
Review this entire project.

It seems like things are implemented at a "shallow" level of things working.

What would I ask about this project if I were to know what to ask so that it's a phenomenally effective, extremely valuable, lightweight + fast, enterprise-grade AI-assistant that fully fulfills the vision of :

"<What Senter Does (The Product)
Senter is a local AI assistant that works for you 24/7 — not just when you're talking to it.
Unlike ChatGPT or Alexa, Senter:

Learns your goals by analyzing your conversations and computer activity
Works autonomously on research, organization, and planning while you're away
Never sends your data to the cloud — everything runs on your hardware

Think of it as a personal executive assistant that lives in your computer, remembering everything you've discussed, organizing your files, researching your interests, and preparing action plans for your review.

How Senter Does It (The Architecture)
The Core Loop: Interact → Route → Respond → Reflect

Interact: You talk, type, or share files. Senter listens via gaze + speech detection (no wake word needed — just look at your camera and talk).
Route: Every input goes through a function calling system that determines what you need. Calendar? Research? Terminal command? Email draft? Senter picks the right tool from an infinitely expandable toolkit.
Respond: Senter responds via text and speech, streaming responses sentence-by-sentence for natural conversation flow. Meanwhile, a parallel research agent is already fetching relevant information.
Reflect: After every interaction, Senter updates:

Your personality profile (preferences, rules, patterns)
Your goal list (things you're working toward)
Your task queue (things Senter can work on autonomously)



The Magic: Dual-Worker Parallel Processing
Senter runs two inference processes simultaneously on your GPUs:

One handles your request
One does background research, journaling, or autonomous tasks

This means Senter is always working, not just responding.

Why Senter Exists (The Vision)
The Problem: AI is powerful, but:

Cloud AI harvests your data for corporate profit
Current assistants are reactive — they wait for you to ask
Expertise is locked behind programming knowledge

The Solution: Senter gives everyone a private, proactive AI that:

Runs the largest open-source models locally on consumer hardware
Learns and works autonomously toward your goals
Protects your data by design — nothing leaves your machine

The Philosophy: "Make your data work for you, not for someone else."
We believe AI should be like electricity — infrastructure for everyone, not a surveillance tool for corporations.

One-Liner Versions
For VCs: "A local AI assistant that works on your goals 24/7, runs the largest open-source models on consumer hardware, and never sends your data to the cloud."
For Consumers: "An AI assistant that actually learns you, works while you're away, and keeps your data private."
For Developers: "An always-on, multi-agent orchestration system with parallel inference, continuous journaling, and privacy-first architecture — all running locally.">"

## Gap Vision_v1
This is a great moment to pause. Let me be honest.

  ---
  Holistic View: What We Built vs. What Senter Promises

  The Vision says:
  "A local AI assistant that works for you 24/7 — not just when you're talking to it."

  What we actually built: A daemon with 64 implemented stories that can be started, stopped, and queried via CLI.

  ---
  Practical Reality Check
  Vision Promise: "Learns your goals by analyzing conversations and computer activity"
  What Exists: GoalTracker with keyword extraction, EventsDB for logging
  Gap: No computer activity analysis. No actual learning - just logging.
  ────────────────────────────────────────
  Vision Promise: "Works autonomously on research, organization, and planning while you're away"
  What Exists: ResearchTaskQueue + Scheduler
  Gap: It queues tasks. But what does it actually do? Sends prompts to Ollama. No real agency.
  ────────────────────────────────────────
  Vision Promise: "Just look and talk" (gaze + speech)
  What Exists: Audio pipeline + Gaze detector
  Gap: Wired up but brittle. No one has demo'd this working smoothly.
  ────────────────────────────────────────
  Vision Promise: "Routes to infinitely expandable toolkit"
  What Exists: ToolRegistry + MCP client
  Gap: Keyword-based routing. Not semantic. MCP servers disabled.
  ────────────────────────────────────────
  Vision Promise: "After every interaction, updates personality/goals/tasks"
  What Exists: PatternDetector, PreferenceLearner
  Gap: It logs things. Doesn't change behavior based on them.
  ────────────────────────────────────────
  Vision Promise: "Dual-worker parallel processing - always working"
  What Exists: Two Ollama workers
  Gap: Different temperature settings. Not parallel thinking. One for user, one for... slightly different prompts.
  ---
  Meta-Perspective: What Went Wrong

  Ralph Wiggums optimized for velocity and test coverage, not value delivery.

  We built:
  - ✓ The skeleton (architecture)
  - ✓ The plumbing (message bus, IPC, queues)
  - ✓ The shell (classes exist, tests pass)

  We didn't build:
  - ✗ The brain (actual intelligence)
  - ✗ The memory (learning that changes behavior)
  - ✗ The agency (true autonomous action)

  Each story was atomic and testable. But "test passes" ≠ "feature delivers value".

  ---
  Meta-Meta-Perspective: The Decomposition Trap

  When you decompose a vision into 64 user stories, you get all the parts but miss the soul.

  The vision says: "An AI that works for you 24/7"

  We built: "A daemon that can call Ollama when triggered"

  If you demo this to someone, would they say "wow, this is working for me 24/7"?

  Honest answer: No. They'd see a CLI with some background processes.

  ---
  What's Actually Missing

  1. No Computer Activity Analysis - Vision mentions it explicitly. We built zero: no file watchers, clipboard monitoring, browser activity, app usage.
  2. No Real Autonomous Work - Scheduler triggers jobs. But what jobs? "Research" = asking Ollama generic questions.
  3. No Real Goal Understanding - Goals are stored as strings. Not understood, prioritized, or progressed toward.
  4. No Real Proactive Behavior - System is fundamentally reactive. Even "autonomous" parts run on timers, not in response to opportunities.
  5. No UI - Where's the menubar app? The notification when Senter learns something? It's a daemon you start via terminal.

  ---
  The Hard Question

  What would make this genuinely useful TODAY?

  1. Forget half the features - Gaze detection, MCP, audio pipeline are premature. Focus on core value.
  2. Make ONE thing great - Pick: autonomous research OR goal tracking OR file organization. Not all three at shallow depth.
  3. Ship something usable - Where's the actual interface? A menubar icon that shows "Senter is researching X for you"?
  4. Measure real value - Did Senter save you 10 minutes today? How? We have no metrics for utility.

  ---
  My Honest Take

  We built a technically impressive scaffolding. 64 stories, comprehensive test coverage, clean architecture.

  But if the question is "Is this a phenomenally effective AI assistant that fulfills the vision?" - the answer is not yet.

  The path forward isn't more stories. It's picking the ONE use case that would make you say "I can't live without this" and making it bulletproof.

  What's that use case for you?

✻ Brewed for 1m 4s

❯ what do you think it should be? 

⏺ Autonomous Research.

  Here's my reasoning:

  ---
  Why Research is the Killer App

  1. It's the only feature that truly needs a daemon.

  Gaze detection? Voice? Those work fine in a foreground app. The only reason to have Senter running 24/7 is if it's doing something when you're not there.

  2. It's the clearest differentiator from ChatGPT.

  ChatGPT waits for you to ask. Senter anticipates and works ahead. You mention "I'm curious about Rust's memory model" in passing. Next morning, there's a research summary waiting.

  3. Privacy actually matters here.

  Researching health symptoms. Career options. Financial strategies. Relationship advice. These are exactly the topics where you don't want cloud AI knowing. Local-first isn't a feature - it's the point.

  4. It's measurable.

  "Senter produced 3 research summaries this week that saved me 2 hours of googling." That's real, quantifiable value.

  5. The pieces already exist.

  We built ResearchTaskQueue, ResearchResultSummarizer, AutonomousResearchSelector, ResearchResultPresenter. They're shallow, but the skeleton is there.

  ---
  What "Research Done Right" Looks Like
  ┌─────────────────────────┬───────────────────────────────────────────────────────────┐
  │      Current State      │                     What It Should Be                     │
  ├─────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Keyword topic detection │ Semantic understanding from conversation context          │
  ├─────────────────────────┼───────────────────────────────────────────────────────────┤
  │ "Research" = ask Ollama │ Actually search the web, read articles, synthesize        │
  ├─────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Store results in JSON   │ Proactive notification: "I found something about X"       │
  ├─────────────────────────┼───────────────────────────────────────────────────────────┤
  │ No feedback loop        │ "Useful" / "Not what I needed" → improves future research │
  └─────────────────────────┴───────────────────────────────────────────────────────────┘
  ---
  The Experience I'd Want

  Monday 10pm: I'm chatting with Senter about a work problem. I mention "I wonder if event sourcing would help here."

  Tuesday 7am: I open my laptop. Menubar icon pulses.

  "While you were away, I researched event sourcing patterns. Found 3 relevant articles and synthesized key points. Want the summary?"

  I click yes. Get a 2-minute read with sources. Click "useful."

  Senter learns: this person cares about architecture patterns. Next time I mention something similar, it's already on it.

  ---
  What Needs to Happen

  1. Real web search - Not Ollama hallucinating. Actual web crawling with sources.
  2. Semantic topic extraction - Use embeddings to understand what I care about from conversation flow, not keyword grep.
  3. Multi-source synthesis - Read 5 articles, synthesize into one coherent summary with citations.
  4. Proactive UI - Menubar app that pulses when research is ready. Not buried in CLI.
  5. Feedback that matters - Thumbs up/down that actually reweights what gets researched.

  ---
  Why Not the Other Features?
  ┌───────────────────┬────────────────────────────────────────────────────────────┐
  │      Feature      │                       Why Not First                        │
  ├───────────────────┼────────────────────────────────────────────────────────────┤
  │ Gaze + Voice      │ Cool demo, but not unique value. Siri/Alexa exist.         │
  ├───────────────────┼────────────────────────────────────────────────────────────┤
  │ Goal tracking     │ Vague. What does "progress toward goal" mean concretely?   │
  ├───────────────────┼────────────────────────────────────────────────────────────┤
  │ File organization │ Dangerous to get wrong. High stakes for low frequency.     │
  ├───────────────────┼────────────────────────────────────────────────────────────┤
  │ Tool routing      │ Infrastructure, not value. Users don't care about routing. │
  └───────────────────┴────────────────────────────────────────────────────────────┘
  ---
  Research is the thing where:
  - The daemon architecture makes sense
  - Privacy is a genuine advantage
  - The value is tangible and measurable
  - You'd actually miss it if it was gone

  Everything else is either premature optimization or a feature looking for a problem.

  What do you think?

✻ Worked for 47s

❯ ok let's build that. I also want you to browse this project "<https://github.com/helloprkr/SENTER-ai/tree/jp-v2>" that was an old version of Senter, but it had a sleek UI that might     
  work... I want to you use that if possible. SO. Make a comprehensive plan, review it to make sure it's building exactly what we want and what will work, and then implement.              

⏺ Let me explore that old Senter repo first to understand the UI, then create a comprehensive plan.

⏺ Fetch(https://github.com/helloprkr/SENTER-ai/tree/jp-v2)
  ⎿  Received 476.6KB (200 OK)

⏺ Fetch(https://github.com/helloprkr/SENTER-ai/tree/jp-v2/components)
  ⎿  Received 181.7KB (200 OK)

⏺ Fetch(https://raw.githubusercontent.com/helloprkr/SENTER-ai/jp-v2/components/liquid-glass-app.tsx)
  ⎿  Error: Request failed with status code 404

⏺ Fetch(https://github.com/helloprkr/SENTER-ai/blob/jp-v2/app/page.tsx)
  ⎿  Received 183KB (200 OK)

⏺ Fetch(https://github.com/helloprkr/SENTER-ai/blob/jp-v2/liquid-glass-app.tsx)
  ⎿  Received 251.5KB (200 OK)

⏺ Fetch(https://github.com/helloprkr/SENTER-ai/blob/jp-v2/app/api/chat/route.ts)
  ⎿  Received 195.5KB (200 OK)

⏺ Search(pattern: "**/web_search*.py", path: "~/Desktop/__2026_/Senter ⎊")
  ⎿  Found 1 file (ctrl+o to expand)

⏺ Search(pattern: "**/search*.py", path: "~/Desktop/__2026_/Senter ⎊")
  ⎿  Found 2 files (ctrl+o to expand)

⏺ Read(Functions/web_search.py)
  ⎿  Read 217 lines

⏺ Search(pattern: "class Research", path: "~/Desktop/__2026_/Senter ⎊", output_mode: "content")
  ⎿  Found 7 lines (ctrl+o to expand)

⏺ Now I have a clear picture. Let me create a comprehensive plan.

⏺ Write(RESEARCH_VISION.md)
  ⎿  Wrote 328 lines to RESEARCH_VISION.md                                                                                                                      
     # Senter: Autonomous Research - Implementation Plan
     ## The Goal
     **One sentence:** When you mention something interesting in conversation, Senter researches it while you're away and has a summary waiting when you return.
     **The experience:**
     ```
     Monday 10pm: You chat with Senter about a work problem. You mention "I wonder if event sourcing would help here."
     … +318 lines (ctrl+o to expand)

⏺ Read(RESEARCH_VISION.md)
  ⎿  Read 328 lines

⏺ I've created RESEARCH_VISION.md with the comprehensive plan. Here's the summary:

  ---
  The Plan at a Glance

  Goal: When you mention something interesting, Senter researches it while you're away and has a summary waiting when you return.

  8 Stories, 3 Phases:

  Phase 1: Research Engine (The Brain)
  ┌────────┬──────────────────────────────────────────────────────────────┐
  │ Story  │                         What It Does                         │
  ├────────┼──────────────────────────────────────────────────────────────┤
  │ RE-001 │ Semantic topic extraction using LLM (not keyword grep)       │
  ├────────┼──────────────────────────────────────────────────────────────┤
  │ RE-002 │ Deep web research: multi-query search + fetch actual content │
  ├────────┼──────────────────────────────────────────────────────────────┤
  │ RE-003 │ Multi-source synthesis with citations                        │
  ├────────┼──────────────────────────────────────────────────────────────┤
  │ RE-004 │ Quality scoring to filter bad research                       │
  └────────┴──────────────────────────────────────────────────────────────┘
  Phase 2: Menubar UI (The Interface)
  ┌────────┬───────────────────────────────────────────────────┐
  │ Story  │                   What It Does                    │
  ├────────┼───────────────────────────────────────────────────┤
  │ UI-001 │ Python menubar app (rumps) with notification      │
  ├────────┼───────────────────────────────────────────────────┤
  │ UI-002 │ Research summary panel with glassmorphism styling │
  ├────────┼───────────────────────────────────────────────────┤
  │ UI-003 │ Feedback buttons (useful/not helpful/more)        │
  └────────┴───────────────────────────────────────────────────┘
  Phase 3: Integration
  ┌─────────────┬─────────────────────────────────────┐
  │    Story    │            What It Does             │
  ├─────────────┼─────────────────────────────────────┤
  │ INT-001-003 │ Wire everything together, polish UX │
  └─────────────┴─────────────────────────────────────┘
  ---
  Key Decisions Made

  1. Python-native UI - rumps for menubar, PyQt6 for panel (not Electron)
  2. Real web research - DuckDuckGo search + httpx + readability extraction
  3. LLM for intelligence - Ollama for topic extraction and synthesis
  4. SQLite for storage - Already in use, simple

  ---
  Questions Before I Start

  1. Frequency: How often to check for topics? (I'm thinking: after every conversation ends)
  2. Depth: How many sources? (I'm thinking: 5-8 articles, synthesize top 4)
  3. Notification: When to alert? (I'm thinking: immediately when research completes)
  4. Glassmorphism priority: Is the sleek UI essential for v1, or can we start with basic native look and add styling in v2?

  ---
  My Recommendation

  Start with the core loop working end-to-end, even if ugly:
  1. RE-001 → RE-002 → RE-003 (research works)
  2. UI-001 (basic menubar notification)
  3. Quick test: does it actually feel useful?
  4. Then add polish (RE-004, UI-002 styling, UI-003 feedback)

  Does this plan look right? Any changes before I start implementing?

✻ Cooked for 3m 53s

❯ 1. Frequency, yes after every convo ends. 2, Depth: yes, 5-8, synth top 4. 3, Notifications: yes. 4: Sleek UI is essential, so let's just do it now, but we can get the core loop working 
  end-to-end even if it's ugly for now. BEFORE you begin, please take a min to maximize and leverage all possible value out of the way the USER STORIES are created within the Ralph        
  Wiggums Skill. Velocity and test passing are important, but only if they bring phenomenal value to the end-result and the user. The Ralph Wiggum skill should help us create              
  revolutionarily valuable experience/end-results with every USER STORY.                                                                                                                    

⏺ You're absolutely right. Let me pause and rethink how we structure these stories.

  ---
  The Problem with How We Did 64 Stories

  We optimized for:
  - Does the code exist? ✓
  - Do tests pass? ✓

  We didn't optimize for:
  - Does this actually deliver value?
  - Would a human say "wow, this works"?

  ---
  New Story Philosophy

  Every story must answer YES to these questions:

  1. Can I demo this to someone? (Not "the class exists" but "watch this work")
  2. Would a user notice if this was missing? (Real value, not plumbing)
  3. Is the acceptance criteria about experience, not implementation?
  4. Is there a quality gate, not just pass/fail?

  ---
  Story Structure That Delivers Value

  Old Pattern (What We Did):
  RE-001: Semantic Topic Extraction
  Acceptance:
  - Extracts topics from text
  - Returns list of topics with priority
  - typecheck passes
  Test: test_topic_extraction.py passes

  New Pattern (What We Should Do):
  RE-001: Senter Notices What You're Curious About

  VALUE STATEMENT:
  After any conversation, Senter identifies what you're genuinely curious
  about—not just keywords, but real interest signals.

  DEMO SCRIPT:
  1. Have a 5-message conversation about a work problem
  2. Mention "I wonder if event sourcing would help here"
  3. End conversation
  4. See: "Detected topic: Event Sourcing (priority: high, reason: expressed curiosity)"

  ACCEPTANCE CRITERIA:
  - Given conversation with clear curiosity signal ("I wonder", "what is", "how does"),
    extracts the topic with priority > 0.7
  - Given conversation with casual mention (no curiosity signal),
    does NOT extract as high priority
  - Given conversation about multiple topics,
    ranks by genuine interest, not mention frequency

  QUALITY GATE (Manual):
  - Run on 10 real test conversations
  - Rate each extraction: Good / Okay / Bad
  - Must achieve: >8 Good, <1 Bad

  TECHNICAL TESTS:
  - Unit tests for extraction logic
  - Integration test: conversation → extracted topics

  ---
  The Key Differences
  ┌────────────────────────────┬──────────────────────────────────────┐
  │        Old Approach        │             New Approach             │
  ├────────────────────────────┼──────────────────────────────────────┤
  │ "Class exists with method" │ "Feature works end-to-end"           │
  ├────────────────────────────┼──────────────────────────────────────┤
  │ "Returns list"             │ "Returns correct list"               │
  ├────────────────────────────┼──────────────────────────────────────┤
  │ "Tests pass"               │ "Tests pass AND quality gate passes" │
  ├────────────────────────────┼──────────────────────────────────────┤
  │ "Given X, when Y, then Z"  │ "Demo: do X, see Y, feel Z"          │
  ├────────────────────────────┼──────────────────────────────────────┤
  │ Implementation-focused     │ Experience-focused                   │

## Gap Vision_v2

### Date: 2026-01-08

---

## The Perfect Senter Experience: What It Should Feel Like

Imagine waking up tomorrow and Senter just... works. Here's what that experience looks like:

### Morning Scenario (The Dream)

**7:00 AM** - You open your laptop. The Senter menubar icon pulses softly.

"While you were away, I researched 3 topics you mentioned yesterday. Event sourcing patterns looks most relevant to your current project. Want the summary?"

You glance at the camera. Senter activates.

"Yes, show me."

A beautiful glass panel slides in with:
- 3 synthesized articles with citations
- Key takeaways in bullet points
- "Useful / Not what I needed / Tell me more" buttons

You click "Useful." Senter learns.

**7:15 AM** - You're working. You copy some code to clipboard.

Senter notices: "I see you're working with async patterns. Want me to research best practices for error handling in async Rust?"

"Yes, queue that."

Task created. You see it in the Tasks panel. Progress bar shows "Researching..."

**12:00 PM** - Notification: "Research complete: Async Rust Error Handling. Found 4 quality sources."

You click the Journal tab. It shows:
- Today's research: 2 completed, 1 in progress
- Topics you're curious about: Event sourcing, Rust async, Memory models
- Patterns detected: "You work on architecture problems in the morning"

**The Feeling**: "I have a personal executive assistant that actually works for me 24/7."

---

## Current Reality vs. Vision (Feature by Feature)

### 1. CHAT

**Vision**: Natural conversation with memory, context, and personality
**Reality**: Text in → Ollama → Text out. No memory across sessions.

| Aspect | Should Be | Actually Is |
|--------|-----------|-------------|
| Memory | Remembers all past conversations | Forgets on restart |
| Context | Includes files, clipboard, sources | Just raw text |
| Personality | Adapts to your style | Generic responses |
| Speed | Streaming sentence-by-sentence | Full response only |

**Gap Score**: 40% complete

### 2. HISTORY

**Vision**: Searchable archive of every conversation, organized by topic
**Reality**: List exists in UI but doesn't load from daemon

| Aspect | Should Be | Actually Is |
|--------|-----------|-------------|
| Persistence | Survives app restart | Lost on restart |
| Search | Find by keyword/date/topic | No search |
| Organization | Grouped by topic/project | Flat list |
| IPC | `get_history` command | Not implemented |

**Gap Score**: 30% complete

### 3. JOURNAL

**Vision**: Daily log of what Senter learned, discovered, and worked on
**Reality**: UI shows "Journal entries coming soon"

| Aspect | Should Be | Actually Is |
|--------|-----------|-------------|
| Backend storage | SQLite table for journal entries | Nothing |
| Auto-generation | Summarizes daily activity | Nothing |
| IPC commands | `get_journal`, `add_journal_entry` | Nothing |
| UI display | Rich entries with timestamps | Placeholder text |

**Gap Score**: 0% complete

### 4. TASKS

**Vision**: Create tasks, watch progress, see results autonomously generated
**Reality**: Task framework exists but TaskExecutor doesn't execute

| Aspect | Should Be | Actually Is |
|--------|-----------|-------------|
| Task creation | "Research X" creates queued task | Can queue, doesn't execute |
| Execution | Actually runs tools, web search | Just logs state changes |
| Progress | Real-time progress bar | No progress tracking |
| Results | Synthesized output with sources | Empty results |
| UI | Task list with status | Placeholder text |

**Gap Score**: 25% complete

### 5. STT (Speech-to-Text)

**Vision**: Look at camera, speak, Senter hears and responds
**Reality**: Audio pipeline code exists but is never called

| Aspect | Should Be | Actually Is |
|--------|-----------|-------------|
| Whisper integration | Working STT engine | Defined but unused |
| Gaze activation | Look → Listen | Gaze is mocked |
| Voice input display | Shows transcription | No voice input |
| Wiring | Gaze → Audio → STT → Query | Completely disconnected |

**Gap Score**: 10% complete (code exists, wiring doesn't)

### 6. WEBCAM / GAZE DETECTION

**Vision**: Ambient awareness - Senter knows when you're looking
**Reality**: Complete MediaPipe implementation, but daemon uses mock

| Aspect | Should Be | Actually Is |
|--------|-----------|-------------|
| Face detection | MediaPipe face mesh | Code exists, unused |
| Attention scoring | face_center + eye_openness + gaze | Mocked with random() |
| Daemon integration | Real camera feed | Random attention signals |
| Privacy controls | Enable/disable camera | Disabled by default |

**Gap Score**: 15% complete (code exists, integration mocked)

### 7. CONTEXT SOURCES

**Vision**: Pin files, URLs, docs as always-relevant context
**Reality**: UI shows empty context menu

| Aspect | Should Be | Actually Is |
|--------|-----------|-------------|
| Add source | Add file/URL/clipboard | No mechanism |
| Storage | Persist sources | No database |
| Retrieval | Include in prompts | Not integrated |
| UI | List with remove/edit | Empty placeholder |

**Gap Score**: 5% complete (UI shell only)

---

## Root Cause Analysis: Why Are We Here?

### 1. The Decomposition Trap
64 user stories were implemented atomically. Each story passed tests. But stories didn't connect to deliver end-to-end value.

**Example**:
- US-001: TaskExecutor class exists ✓
- US-002: Task queue works ✓
- US-003: Task results storage works ✓
- **Reality**: No story said "task actually executes and produces real output"

### 2. Velocity Over Value
Ralph Wiggums optimized for story completion velocity. "Does test pass?" = done.
But "test passes" ≠ "feature works" ≠ "user gets value".

### 3. Missing Integration Layer
Components exist in isolation:
- Gaze detector: Complete code, never instantiated
- Audio pipeline: Complete code, never called
- Task executor: Complete interface, empty implementation

### 4. IPC Gap
- Backend has 11 commands
- Frontend only calls 4
- Missing: history, journal, tasks, context, audio control

### 5. No End-to-End Testing
We tested units. We never tested: "Can a user actually accomplish X?"

---

## The Path Forward: What Needs to Happen

### Critical Path (Must Fix)

1. **Wire the Disconnected Components**
   - Gaze detector → Audio pipeline → STT → Query flow
   - Task executor → Actually execute → Store results
   - Journal generator → Auto-summarize → Display

2. **Complete the IPC Bridge**
   - Add missing handlers: `get_history`, `get_journal`, `get_tasks`, `add_context_source`
   - Frontend must be able to access all daemon data

3. **Persistence Everywhere**
   - Conversations must survive restart
   - History must load from daemon
   - Journal must generate and store daily

4. **Real Task Execution**
   - TaskExecutor must invoke tools
   - Web search must actually search
   - Results must synthesize with citations

### What "Done" Looks Like

**Chat**: You type, Senter responds with context from your history, sources, and personality preferences. Responses stream sentence-by-sentence.

**History**: You search "Rust" and find all conversations mentioning Rust, with relevant excerpts highlighted.

**Journal**: You read today's journal: "Researched 3 topics. You seem interested in async patterns. Completed 2 tasks."

**Tasks**: You say "Research GraphQL best practices." Task appears. Progress shows. 20 minutes later: "Research complete" with 5 sources synthesized.

**Voice**: You look at the camera. "What did you find about GraphQL?" Senter speaks the summary.

**Context**: You pin your project README. Every conversation about the project includes that context.

---

## Prioritized Implementation Plan

### Phase 1: Core Loop (The Foundation)
Make ONE thing work perfectly: **Ask question → Get answer with context → Remember forever**

1. Fix conversation persistence (load/save from daemon)
2. Add conversation context to prompts
3. Stream responses
4. History search

### Phase 2: Autonomous Research (The Differentiator)
Make Senter work while you're away

1. Wire task execution to actually run
2. Real web search (not just Ollama)
3. Multi-source synthesis
4. Notification when complete
5. Journal auto-generation

### Phase 3: Ambient Interaction (The Magic)
"Just look and talk"

1. Wire gaze detector (real, not mock)
2. Wire audio pipeline
3. Gaze → Audio → STT → Query flow
4. TTS responses

### Phase 4: Deep Personalization (The Relationship)
Senter that knows you

1. Context sources (files, URLs)
2. Preference learning that affects responses
3. Proactive suggestions
4. Goal tracking with progress

---

## What Perfect Feels Like

When this is done, using Senter will feel like having a brilliant assistant who:

- **Never forgets** - Everything you've discussed is searchable
- **Works overnight** - Research appears when you wake up
- **Knows you** - Understands your projects, interests, style
- **Requires no typing** - Just look and talk
- **Stays private** - Nothing leaves your machine
- **Gets better** - Learns from every interaction

The value proposition becomes undeniable: "I can't go back to ChatGPT. Senter actually works for me."

---

## Technical Debt to Address

1. **Mocked gaze detection** - Replace random() with real MediaPipe
2. **Audio pipeline disconnected** - Wire to message bus
3. **TaskExecutor empty** - Implement actual execution
4. **4 of 11 IPC handlers** - Add missing 7
5. **No streaming** - Add SSE or chunked responses
6. **Scheduler doesn't trigger** - Implement cron logic
7. **Journal non-existent** - Create from scratch

---

## Success Metrics

How do we know we've succeeded?

1. **Chat latency** < 500ms to first token
2. **History persistence** - Survives 10 restarts, all conversations intact
3. **Task completion** - 90% of research tasks produce useful output
4. **Voice activation** - Gaze detection works 95% of the time
5. **Journal accuracy** - Daily summary matches actual activity
6. **User value** - "Senter saved me 30 minutes today"

---

## Conclusion

Senter is ~40% complete. The architecture is solid. The components exist. But they're not wired together, and "features" are often just infrastructure.

The path forward is NOT more user stories. It's:
1. Wire existing components together
2. Fill in the empty implementations
3. Test end-to-end flows, not units
4. Measure real user value, not test coverage

The vision is achievable. The pieces are here. We just need to connect them.

---

## Gap Vision v2.1 - Implementation Progress (2026-01-08)

### What Was Accomplished

Following the Gap Vision v2 plan, 11 user stories were implemented across 3 phases, bringing Senter from ~40% to ~75% complete.

### Phase 1: Core Loop (100% Complete)

| Story | Title | Status | Key Implementation |
|-------|-------|--------|-------------------|
| P1-001 | Conversation Persistence | ✅ PASSED | `get_conversations` + `save_conversation` IPC handlers, auto-save on bot response |
| P1-002 | History Tab Shows Real Conversations | ✅ PASSED | Click-to-load, active state highlight, New Conversation button |
| P1-003 | Conversation Context in Prompts | ✅ PASSED | `_build_conversation_context` loads last 20 messages, passes to model |
| P1-004 | History Search | ✅ PASSED | Real-time filtering by title and message content |

**Result**: Conversations now persist across restarts. History is searchable. Context flows through to LLM.

### Phase 2: Autonomous Research (100% Complete)

| Story | Title | Status | Key Implementation |
|-------|-------|--------|-------------------|
| P2-001 | Tasks Tab Shows Real Tasks | ✅ PASSED | `get_tasks` IPC handler, status badges, detail view |
| P2-002 | Create Task from Chat | ✅ PASSED | `_detect_research_intent` with regex patterns, creates task from chat |
| P2-003 | Real Task Execution | ✅ PASSED | DuckDuckGo search + httpx content fetching + LLM synthesis |
| P2-004 | Journal Auto-Generation | ✅ PASSED | `JournalDB` with SQLite, daily summaries with topics/counts |
| P2-005 | Desktop Notifications | ✅ PASSED | `TaskPoller` polls for completions, macOS notifications |

**Result**: Research tasks execute with real web search. Journal entries auto-generate. Notifications work.

### Phase 3: Context Sources (100% Complete)

| Story | Title | Status | Key Implementation |
|-------|-------|--------|-------------------|
| P3-001 | Context Sources - Add and Display | ✅ PASSED | `ContextSourcesDB` SQLite, add/remove UI in ContextMenu |
| P3-002 | Context Sources in Query | ✅ PASSED | `_build_context_sources_prompt` includes sources in system prompt |

**Result**: Users can pin files/URLs as context. Sources are included in every query.

### Phase 4: Voice Interaction (0% Complete - In Progress)

| Story | Title | Status | Blocker |
|-------|-------|--------|---------|
| P4-001 | Voice Input Display | ❌ PENDING | Requires audio pipeline wiring to Whisper |
| P4-002 | Real Gaze Detection | ❌ PENDING | Requires camera permissions, MediaPipe integration |
| P4-003 | Gaze Activates Voice | ❌ PENDING | Depends on P4-001 and P4-002 |

### Updated Gap Scores

| Feature | Before | After | Notes |
|---------|--------|-------|-------|
| Chat | 40% | 75% | Context works, streaming still TODO |
| History | 30% | 90% | Persistence + search complete |
| Journal | 0% | 85% | Auto-generation works |
| Tasks | 25% | 85% | Real execution with web search |
| STT | 10% | 10% | No progress yet |
| Gaze | 15% | 15% | No progress yet |
| Context | 5% | 90% | Full CRUD + query integration |

### Key Files Created/Modified

**Python Backend:**
- `learning/journal_db.py` - NEW: SQLite storage for journal entries
- `learning/context_sources_db.py` - NEW: SQLite storage for context sources
- `daemon/ipc_server.py` - EXTENDED: 10 new IPC handlers
- `engine/task_engine.py` - EXTENDED: Real web search with content fetching

**Electron Frontend:**
- `src/main/ipc/senterBridge.ts` - EXTENDED: TaskPoller for notifications
- `src/renderer/components/ChatView/RightPanel.tsx` - REWRITTEN: Real data display
- `src/renderer/components/ContextMenu/ContextMenu.tsx` - REWRITTEN: Add/remove sources
- `src/renderer/store/contextSlice.ts` - EXTENDED: Daemon sync

### What's Still Missing for Full Vision

1. **Voice Interaction** - Phase 4 stories (in progress)
2. **Streaming Responses** - Sentence-by-sentence output
3. **Semantic Topic Extraction** - Currently using regex patterns
4. **Feedback Loop** - Thumbs up/down that affects future research
5. **Computer Activity Analysis** - File watchers, clipboard monitoring

### Next Steps

1. Complete Phase 4 (Voice Interaction)
2. Add response streaming
3. Implement feedback-driven learning
4. Add semantic embeddings for smarter topic detection

---

