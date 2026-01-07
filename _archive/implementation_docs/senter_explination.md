What Senter Does (The Product)
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
For Developers: "An always-on, multi-agent orchestration system with parallel inference, continuous journaling, and privacy-first architecture — all running locally."


## SENTER described by Chris from chats:
SUPPLEMENTARY OUTPUT 1: Everything Known About Senter
Core Identity

Name: Senter (stylized as "Center" — "your AI center, data center, idea center")
Vision: Always-on autonomous AI assistant that continuously works toward user's goals
Philosophy: Privacy-first, local-first, "make your data work for you, not for someone else"
Target: Consumer AI workstation market

Architecture Components
ComponentFunctionStatusAVA (Attentive Voice Assistant)Gaze + speech detection for attention-based wake (no wake word)Functional on new hardwareNeural CenterCore orchestrator — dual-worker parallel processingFunctionalCenter 1B ScriptsTwin inference (research + function calling simultaneously)FunctionalUniversal Response AgentCompiles system prompt from journal notes, personality, rulesExistsAgent Similarity.pySimilarity search for top-K tool selectionFunctional, "pretty damn accurate"Journal/Reflection SystemExtracts personality notes, workflow rules, generates task list"Weakest part" — needs workTask List (Center Goals)Perpetual list including "wander" tasks for autonomous explorationConceptual, partial implementationFunction Calling ConfigInfinitely expandable agent list in XML formatWorking
Input/Output Modalities

Input: Text, voice, files, images, video, multimodal
Output: Text, speech (Piper TTS), streaming generation with per-sentence processing
Wake: Gaze detection + speech (no wake word)
Access: CLI (carrot commands), future UI, double-tap control on hardware

Technical Stack

Llama CPP Python for inference
Mistral Small / Devstrol for final response + vision
Hermes 8B for internal function calling
Piper for TTS (trainable)
Quen Omni 3B for video processing
Nomic embed for embeddings
Newell (GNU) integration for clean OS-level pipelines
Compatible with AGENT.md format (Claude Code interop)

Key Behaviors

Interact → Route → Respond → Reflect (core loop)
Every prompt triggers parallel research + function calling
Journal continuously analyzes conversation for goals/preferences
Task list updated with user-approved and autonomous items
"Wander" tasks generate related search queries based on interests
Personality mirroring through journal-injected system prompts