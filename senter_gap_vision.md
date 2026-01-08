# Senter Gap Vision

> "A local AI assistant that works for you 24/7 — not just when you're talking to it."

---

## The Vision (What We're Building Toward)

**For VCs:** "A local AI assistant that works on your goals 24/7, runs the largest open-source models on consumer hardware, and never sends your data to the cloud."

**For Consumers:** "An AI assistant that actually learns you, works while you're away, and keeps your data private."

**For Developers:** "An always-on, multi-agent orchestration system with parallel inference, continuous journaling, and privacy-first architecture — all running locally."

### Core Product Features (The WOW Moments)
1. **Learns your goals** by analyzing conversations and computer activity
2. **Works autonomously** on research, organization, and planning while you're away
3. **Never sends data to the cloud** — everything runs on your hardware
4. **No wake word** — just look at your camera and talk (gaze + speech detection)
5. **Parallel processing** — one worker handles requests, one does background work
6. **Continuous reflection** — updates your profile, goals, and task queue after every interaction

---

# Gap Vision V1 - COMPLETE

## Started: 2026-01-07
## Completed: 2026-01-08

### What Was Built (Technical)
- Core engine with memory systems (semantic, episodic, procedural, affective)
- Goal detection from conversations
- Activity monitoring (screen context, window detection)
- Trust-based evolution system
- Proactive suggestion engine
- HTTP API server (daemon/http_server.py)
- Electron + Next.js UI with glassmorphism design
- 335 passing tests

### What Was Delivered (User Value)
| Feature | Status | User Can... |
|---------|--------|-------------|
| Chat with Senter | Partial | Have a conversation (but responses are basic) |
| See detected goals | UI Ready | See goals panel (but no real goal detection happening) |
| Activity monitoring | Backend Ready | Backend detects context (but not visible to user) |
| Proactive suggestions | UI Ready | See suggestion toasts (but no real suggestions generated) |
| Welcome back summary | UI Ready | See away modal (but no actual research done while away) |

### Gap Analysis V1

**The Problem:** Infrastructure is built, but nothing is actually RUNNING.

| Vision Feature | Implementation | Gap |
|----------------|----------------|-----|
| "Learns your goals" | GoalDetector class exists | Never called during real conversations |
| "Works while you're away" | auto_research_learning_goals() exists | Never scheduled to run |
| "Activity monitoring" | ActivityMonitor class exists | Not wired to daemon properly |
| "Proactive suggestions" | ProactiveSuggestionEngine exists | Only generates when trust > 0.6 (never reached) |
| "Parallel processing" | Two-worker architecture designed | Only one worker actually runs |

### Key Findings

1. **The app starts but doesn't DO anything valuable**
   - You can type, but responses don't learn from you
   - Goals panel exists but shows nothing (no detection running)
   - Activity badge exists but shows "unknown" (monitor not active)

2. **Trust system is a chicken-and-egg problem**
   - Proactive features require trust > 0.6
   - Trust only increases through successful interactions
   - But interactions aren't successful because features aren't working

3. **Background processing never happens**
   - Daemon exists but only serves HTTP
   - No actual background research/learning loop
   - "Works while you're away" is a lie currently

4. **The UI is beautiful but empty**
   - Glassmorphism design looks great
   - But it's showing placeholder/empty states
   - No real data flowing through

### V1 Resolution Status
- [x] OpenAI dependency installed
- [x] HTTP server imports correctly
- [x] All 335 tests pass
- [x] UI builds and compiles
- [ ] **NOT VERIFIED: Does it actually deliver value to a user?**

---

# Gap Vision V2 - USER VALUE FOCUS

## Goal: Make Senter Actually DO Something Valuable

### User Profile
- **Who:** Knowledge worker, developer, or creative professional
- **Core Problem:** Wants an AI that learns them, remembers context, and works proactively
- **WOW Moment:** Opening the app and seeing Senter already knows what they're working on

### User Stories (Value-First)

#### US-V2-001: User can have a conversation Senter remembers
**Value:** Conversation history persists and influences future responses
**WOW:** Ask "what were we talking about yesterday?" and get an answer
**Acceptance:**
- [ ] Previous conversations are loaded on startup
- [ ] Senter references past discussions naturally
- [ ] Memory feels seamless, not like a transcript dump

#### US-V2-002: User sees their AI-detected goals when they open the app
**Value:** Without entering anything, Senter shows what it thinks user is working toward
**WOW:** "How did it know I was working on that?"
**Acceptance:**
- [ ] Goals appear on first load
- [ ] At least one goal is detected from conversation history
- [ ] Goals have progress indicators that feel meaningful
- [ ] User can correct/dismiss goals

#### US-V2-003: User gets proactive suggestions that feel relevant
**Value:** Senter offers help before being asked
**WOW:** Suggestion appears right when user needed it
**Acceptance:**
- [ ] Suggestions appear as non-intrusive toasts
- [ ] Suggestions are based on detected activity/goals
- [ ] At least 50% of suggestions should feel useful
- [ ] Easy to dismiss or act on

#### US-V2-004: User sees what Senter learned while they were away
**Value:** After time away, user gets a summary of what Senter did
**WOW:** Senter actually researched something relevant
**Acceptance:**
- [ ] Away modal appears when user returns (>30 min away)
- [ ] Summary includes at least one piece of new information
- [ ] Research is related to user's actual goals
- [ ] Sources are cited

#### US-V2-005: User can see their activity context recognized
**Value:** Senter knows "you're coding" or "you're browsing"
**WOW:** Context-aware responses ("I see you're in VS Code...")
**Acceptance:**
- [ ] Activity badge shows current context
- [ ] Context updates in real-time (within 30 seconds)
- [ ] At least 3 context types work (coding, browsing, writing)
- [ ] Context influences conversation responses

#### US-V2-006: Conversation responses feel personalized
**Value:** Senter's tone and suggestions match user preferences
**WOW:** "It talks like I like to be talked to"
**Acceptance:**
- [ ] Response style adapts over time
- [ ] Preferences are persisted
- [ ] User can explicitly set preferences
- [ ] Difference is noticeable after 5+ conversations

### Technical Prerequisites (Behind the Scenes)

Before value stories can be delivered:
1. [ ] Daemon must actually RUN (not just serve HTTP)
2. [ ] Activity monitor must be wired and collecting data
3. [ ] Goal detection must run on every conversation
4. [ ] Background research loop must be scheduled
5. [ ] Trust must bootstrap from 0 (or lower threshold for features)

### Success Criteria for V2

V2 is COMPLETE when a new user can:
1. Start the app and have a conversation
2. Close the app, come back later, and Senter remembers
3. See at least one AI-detected goal they didn't enter
4. Receive at least one proactive suggestion that's useful
5. See their activity context recognized correctly
6. Feel like Senter is "learning" them over time

---

# Gap Vision V3+ (Future)

Topics for future gap analyses:
- Voice interaction (gaze + speech detection)
- Dual-worker parallel processing
- File organization capabilities
- Calendar/email integration
- Multi-modal inputs (images, documents)
- Enterprise-grade security audit
- Performance optimization for consumer hardware

---

## Methodology

Each Gap Vision cycle:
1. **Assess Current State** — What exists vs. what works
2. **Define User Value Stories** — What can user DO after this cycle?
3. **Implement with Ralph Wiggums v2.0** — Value-first autonomous development
4. **Verify Value Delivered** — Actually use the product, not just run tests
5. **Document Findings** — Update this file for next cycle
