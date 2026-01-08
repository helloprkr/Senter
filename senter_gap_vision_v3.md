# Senter Gap Vision v3: The Final Mile

**Date:** 2026-01-08
**Previous:** Gap Vision v2.1 (14/14 stories complete)
**Status:** Deep re-analysis after comprehensive codebase exploration

---

## Executive Summary

After a thorough codebase exploration, the reality is **surprisingly positive**. The previous gap analyses (v1, v2) underestimated how much infrastructure was actually complete. The daemon has **53 IPC handlers**, a **complete research pipeline**, **4 learning databases**, and **full voice/gaze code**.

**The Gap is NOT "things don't exist." The Gap is "things exist but aren't activated."**

| What We Thought | What's Actually True |
|-----------------|----------------------|
| Research is stub code | 4-stage pipeline is complete (topic → search → synthesis → storage) |
| Learning doesn't work | Pattern detection runs, just doesn't affect behavior |
| Voice/Gaze disabled | Code complete, config-driven, optional dependencies graceful |
| IPC is incomplete | 53 handlers fully implemented |
| Task execution is empty | Full task engine with dependencies, retries, persistence |

---

## The REAL Gaps (What's Actually Missing)

### Gap 1: Auto-Research Trigger (Critical)
**What exists:** Complete research pipeline - TopicExtractor, DeepResearcher, Synthesizer, ResearchStore
**What's missing:** Nothing triggers it automatically

The research pipeline can:
- Extract topics from conversation with priority scoring
- Search web with multiple query angles
- Fetch actual page content with readability extraction
- Synthesize with LLM and add citations
- Store with feedback ratings

But the trigger is manual. No automatic detection of "user expressed curiosity → queue research."

**Impact:** The "research while you're away" vision doesn't happen automatically.

### Gap 2: Learning Doesn't Affect Behavior (Critical)
**What exists:**
- EventsDB storing all interactions
- PatternDetector analyzing peak hours, topics, preferences
- BehaviorAnalyzer extracting patterns
- Preferences stored with confidence scores

**What's missing:** None of this changes how Senter responds.

The pattern detector knows:
- Your peak productivity hours
- Topics you ask about most
- Your response length preferences
- Your communication style

But model workers don't query this. System prompts don't include preferences.

**Impact:** After 1000 conversations, Senter behaves exactly the same as conversation 1.

### Gap 3: Streaming Responses (UX Critical)
**What exists:** Model workers return full responses
**What's missing:** Sentence-by-sentence streaming

Every response waits for complete generation. No interim display.

**Impact:** Long responses feel slow. No natural conversation flow.

### Gap 4: Proactive Notifications (Vision Critical)
**What exists:**
- Reporter generates activity digests
- Notification code in Electron (showResearchComplete)
- TaskPoller checks for completions

**What's missing:** Proactive "I noticed something" notifications

System only notifies when explicitly requested (research complete). Doesn't proactively suggest:
- "You mentioned X yesterday, I researched it"
- "Based on your patterns, you might want to..."
- "I noticed you're working on Y, here's context"

**Impact:** User doesn't know Senter is "working" unless they check.

### Gap 5: UI Doesn't Surface Intelligence (UX Gap)
**What exists:**
- Context sources database and IPC handlers
- Goals database with embeddings
- Pattern detection results
- Learning insights storage

**What's missing:** UI doesn't show any of this

The UI has ChatView, History, Journal, Tasks - but doesn't show:
- Your detected patterns ("You're most active 9-11am")
- Your learned preferences ("You prefer detailed responses")
- Your goals and progress
- Proactive research suggestions

**Impact:** User can't see what Senter "knows" about them.

### Gap 6: Worker Model Differentiation (Optimization)
**What exists:** Two model workers with different system prompts
**What's missing:** Actually different models or capabilities

Both workers use `llama3.2` with same parameters. The "research worker" is just primary with different prompt.

**Impact:** No specialized research capability, just rebranding.

---

## What's Already Complete (No Action Needed)

| Component | Status | Evidence |
|-----------|--------|----------|
| IPC Server | ✅ COMPLETE | 53 handlers, socket at /tmp/senter.sock |
| Message Bus | ✅ COMPLETE | Pub/sub, DLQ, correlation tracking |
| Task Engine | ✅ COMPLETE | Full execution, dependencies, retries |
| Research Pipeline | ✅ COMPLETE | 4-stage, web search, synthesis |
| Learning DBs | ✅ COMPLETE | Events, behavior, journal, context |
| Pattern Detection | ✅ COMPLETE | Peak hours, topics, preferences |
| Audio Pipeline | ✅ COMPLETE | VAD, STT, TTS (optional) |
| Gaze Detection | ✅ COMPLETE | MediaPipe, attention (optional) |
| UI Components | ✅ COMPLETE | Chat, Sidebar, Settings |
| Electron IPC | ✅ COMPLETE | All 20+ methods wired |

---

## The v3 Plan: Activate What Exists

Instead of building new features, we need to **wire existing capabilities**.

### Phase 1: Auto-Research Activation
Make the research pipeline trigger automatically when curiosity is detected.

| Story | Title | What Changes |
|-------|-------|--------------|
| V3-001 | Curiosity Detection in Chat | After each response, check for research triggers |
| V3-002 | Research Queue Auto-Population | Detected topics go to research queue |
| V3-003 | Research Completion Notification | Proactive "research ready" notification |

### Phase 2: Learning Personalization
Make detected patterns affect actual responses.

| Story | Title | What Changes |
|-------|-------|--------------|
| V3-004 | Preference Injection | Add learned preferences to system prompt |
| V3-005 | Adaptive Response Style | Adjust verbosity/formality from patterns |
| V3-006 | Goal-Aware Context | Include active goals in prompts |

### Phase 3: Response Streaming
Make responses feel natural and fast.

| Story | Title | What Changes |
|-------|-------|--------------|
| V3-007 | Sentence-by-Sentence Streaming | Stream partial responses to UI |
| V3-008 | Typing Indicator Accuracy | Show real progress, not fake indicator |

### Phase 4: UI Intelligence Visibility
Show users what Senter knows about them.

| Story | Title | What Changes |
|-------|-------|--------------|
| V3-009 | Insights Panel | Show detected patterns in UI |
| V3-010 | Goals Dashboard | Show goals and progress |
| V3-011 | Context Sources Panel | Make pinned sources visible |
| V3-012 | Research Suggestions | Proactive "research this?" prompts |

### Phase 5: Proactive Behavior
Make Senter suggest things before being asked.

| Story | Title | What Changes |
|-------|-------|--------------|
| V3-013 | Morning Digest Notification | "Here's what I worked on overnight" |
| V3-014 | Context-Aware Suggestions | "I noticed you're working on X" |
| V3-015 | Research Proactivity | Auto-queue from conversation, not just explicit requests |

---

## Success Metrics for v3

| Metric | Current | Target |
|--------|---------|--------|
| Auto-research triggers per day | 0 | 3-5 |
| Personalization in responses | 0% | 80%+ |
| Response streaming | No | Yes |
| Patterns visible in UI | No | Yes |
| Proactive notifications per day | 0 | 2-3 |
| User knowing Senter's capabilities | Low | High |

---

## Architecture: What Needs Wiring

```
CURRENT STATE (disconnected):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Conversation │     │ TopicExtract │     │ Research     │
│ Handler      │     │ (complete)   │     │ Pipeline     │
└──────────────┘     └──────────────┘     │ (complete)   │
       │                    ▲              └──────────────┘
       │                    │ NOT WIRED
       ▼                    │
┌──────────────┐     ┌──────────────┐
│ Model Worker │     │ Pattern      │
│ (responds)   │     │ Detector     │
└──────────────┘     │ (complete)   │
                     └──────────────┘
                            │
                     NOT USED ▼
                     ┌──────────────┐
                     │ Preferences  │
                     │ (stored)     │
                     └──────────────┘

TARGET STATE (connected):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Conversation │────▶│ TopicExtract │────▶│ Research     │
│ Handler      │     │ (complete)   │     │ Pipeline     │
└──────────────┘     └──────────────┘     └──────┬───────┘
       │                                         │
       ▼                                         ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Model Worker │◀────│ Preference   │     │ Notification │
│ (personalized)     │ Injector     │     │ System       │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │ Pattern      │
                     │ Detector     │
                     └──────────────┘
```

---

## Implementation Priority

### Must Have (v3.0)
1. V3-001: Curiosity Detection → Research Trigger
2. V3-004: Preference Injection in Prompts
3. V3-003: Research Completion Notifications
4. V3-009: Insights Panel in UI

### Should Have (v3.1)
5. V3-007: Response Streaming
6. V3-005: Adaptive Response Style
7. V3-010: Goals Dashboard

### Nice to Have (v3.2)
8. V3-013: Morning Digest
9. V3-014: Context-Aware Suggestions
10. V3-015: Auto Research from Conversation

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Topic extraction too noisy | Medium | High | Start conservative, require strong signals |
| Preference injection degrades quality | Medium | Medium | A/B test with/without |
| Streaming complexity | Low | Medium | Use existing IPC, just add events |
| UI overload | Medium | Medium | Progressive disclosure, not all at once |

---

## The Philosophy Shift

**v1-v2 Mindset:** "We need to BUILD more features"
**v3 Mindset:** "We need to WIRE existing features"

The codebase has:
- A complete research pipeline sitting idle
- A pattern detector generating unused insights
- Learning databases collecting dust
- Voice/gaze code waiting for activation

The work isn't creation. It's connection.

---

## Conclusion

Gap Vision v3 is about the **final mile** - not building new capabilities, but activating the ones that already exist. The daemon has ~85% of what's needed. The remaining 15% is wiring, not writing.

After v3:
- Senter automatically researches topics you're curious about
- Responses are personalized based on learned patterns
- Users can see what Senter knows about them
- Proactive notifications make Senter feel "alive"

The vision of "an AI that works for you 24/7" becomes real - not by building more, but by connecting what's already there.
