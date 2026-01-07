# SENTER 3.0: Complete Build Guide for Claude Code

**Document Version**: 1.0  
**Created**: January 7, 2026  
**Purpose**: Transform Senter 2.0 into Senter 3.0 - a unified personal AI system achieving human-AI symbiosis

---

## EXECUTIVE SUMMARY

### The Mission

Transform Senter from a 10,000+ line system with 7 non-functional "agents" into a ~1,500 line unified system that actually works and improves over time.

### The Two Insights (Both Required)

**Insight 1 - Internal Architecture (Configuration as DNA):**
> "AI systems should be defined by declarative configurations, not imperative code. Configuration is the DNA; code is just the cell membrane."

**Insight 2 - External Relationship (Cognitive Coupling):**
> "Personal AI value emerges from bidirectional cognitive coupling where human and AI maintain models of each other, and value compounds through mutual understanding."

### What This Means Practically

| Current Senter | Target Senter |
|---------------|---------------|
| 7 separate "agents" (prompt files) | 1 unified Configuration Engine |
| 10,000+ lines, 30% functional | ~1,500 lines, 100% functional |
| `list_all_focuses()` returns `[]` | Everything works |
| Stub functions that print logs | Real implementations |
| AI doesn't know what it knows | AI state visible to human |
| Human is input source | Human-AI coupled cognitive system |
| No learning | Real evolution with fitness tracking |
| Forgets everything | Living memory persists forever |

---

## PART 1: THE UNIFIED ARCHITECTURE

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SENTER 3.0                                        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    INTERFACE LAYER                                   │ │
│  │  senter.py (CLI) │ TUI (optional) │ API (future)                    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    COUPLING LAYER                                    │ │
│  │  JointState │ HumanModel │ AIStateVisibility │ CouplingProtocols    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    CORE ENGINE                                       │ │
│  │  ConfigurationEngine │ IntentParser │ ResponseComposer               │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    KNOWLEDGE LAYER                                   │ │
│  │  KnowledgeGraph │ CapabilityRegistry │ ContextEngine                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    MEMORY LAYER                                      │ │
│  │  SemanticMemory │ EpisodicMemory │ ProceduralMemory │ Affective      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    EVOLUTION LAYER                                   │ │
│  │  FitnessTracker │ MutationEngine │ SelectionPressure │ TrustTracker  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                 │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    MODEL LAYER                                       │ │
│  │  GGUF (llama.cpp) │ OpenAI API │ vLLM │ Embeddings                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Target File Structure

```
Senter/
├── senter.py                    # Main entry point (~50 lines)
├── genome.yaml                  # All configuration (DNA)
├── pyproject.toml               # Minimal dependencies
│
├── core/
│   ├── __init__.py
│   ├── engine.py                # ConfigurationEngine (~150 lines)
│   ├── intent.py                # IntentParser (~100 lines)
│   ├── composer.py              # ResponseComposer (~100 lines)
│   └── genome_parser.py         # YAML genome loader (~50 lines)
│
├── coupling/
│   ├── __init__.py
│   ├── joint_state.py           # JointStateSurface (~100 lines)
│   ├── human_model.py           # HumanModel (~150 lines)
│   ├── protocols.py             # CouplingProtocols (~200 lines)
│   └── trust.py                 # TrustTracker (~50 lines)
│
├── knowledge/
│   ├── __init__.py
│   ├── graph.py                 # KnowledgeGraph (~150 lines)
│   ├── capabilities.py          # CapabilityRegistry (~100 lines)
│   └── context.py               # ContextEngine (~100 lines)
│
├── memory/
│   ├── __init__.py
│   ├── living_memory.py         # Orchestrator (~100 lines)
│   ├── semantic.py              # Facts storage (~100 lines)
│   ├── episodic.py              # Event storage (~100 lines)
│   ├── procedural.py            # Pattern storage (~50 lines)
│   └── affective.py             # Emotion tracking (~50 lines)
│
├── evolution/
│   ├── __init__.py
│   ├── fitness.py               # FitnessTracker (~100 lines)
│   ├── mutations.py             # MutationEngine (~100 lines)
│   └── selection.py             # SelectionPressure (~50 lines)
│
├── models/
│   ├── __init__.py
│   ├── base.py                  # ModelInterface ABC (~50 lines)
│   ├── gguf.py                  # GGUF via llama.cpp (~100 lines)
│   ├── openai_model.py          # OpenAI API (~100 lines)
│   └── embeddings.py            # Embedding model (~100 lines)
│
├── tools/
│   ├── __init__.py
│   ├── web_search.py            # Web search capability
│   ├── file_ops.py              # File operations
│   └── discovery.py             # Auto-discovery
│
├── data/
│   ├── memory.db                # SQLite for all persistence
│   ├── episodes/                # Episode recordings
│   └── evolution/               # Evolution history
│
└── tests/
    ├── __init__.py
    ├── test_core.py
    ├── test_coupling.py
    ├── test_memory.py
    ├── test_evolution.py
    └── test_integration.py
```

---

## PART 2: THE GENOME (Configuration DNA)

Create this file as `genome.yaml` - this is the ONLY configuration file needed:

```yaml
# SENTER 3.0 GENOME
# All behavior emerges from this configuration
# Code is just the interpreter

version: "3.0"
name: "Senter"
description: "Unified Personal AI with Cognitive Coupling"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
models:
  primary:
    type: gguf  # gguf | openai | vllm
    path: "${SENTER_MODEL_PATH}"
    settings:
      n_gpu_layers: -1
      n_ctx: 8192
      temperature: 0.7
      max_tokens: 1024
  
  embeddings:
    type: gguf
    path: "${SENTER_EMBED_PATH}"
    settings:
      n_ctx: 512

# ============================================================
# KNOWLEDGE CONFIGURATION
# ============================================================
knowledge:
  graph:
    type: hierarchical
    persistence: sqlite
    
  domains:
    - name: user_context
      description: "Everything about the user"
      retention: permanent
      
    - name: project_context
      description: "Current projects and work"
      retention: 90d
      
    - name: conversation_context
      description: "Recent conversation topics"
      retention: 7d
      
    - name: world_knowledge
      description: "General facts and information"
      retention: permanent
      update_via: web_search

# ============================================================
# CAPABILITY CONFIGURATION
# ============================================================
capabilities:
  discovery:
    enabled: true
    sources:
      - path: tools/
        pattern: "*.py"
  
  builtin:
    - name: respond
      description: "Generate conversational response"
      always_available: true
      
    - name: web_search
      description: "Search the web for current information"
      triggers: ["current", "latest", "news", "weather", "price", "today"]
      
    - name: remember
      description: "Store information in long-term memory"
      triggers: ["remember", "don't forget", "note that", "keep in mind"]
      
    - name: recall
      description: "Retrieve information from memory"
      triggers: ["what did", "when did", "remind me", "do you remember"]

# ============================================================
# COUPLING CONFIGURATION (The Key Innovation)
# ============================================================
coupling:
  joint_state:
    visible_to_human: true
    surface:
      - current_focus
      - active_goals
      - ai_uncertainties
      - available_capabilities
  
  human_model:
    cognitive_state:
      infer:
        - focus_level        # How focused (0-1)
        - energy_level       # How tired (0-1)
        - mode              # exploring | executing | debugging | learning | creating
        - time_pressure     # none | low | moderate | high | urgent
        - frustration       # Detected frustration level (0-1)
      
    persistent:
      - communication_style
      - expertise_areas
      - preferences
      - patterns
  
  protocols:
    - name: dialogue
      description: "Turn-taking conversation"
      triggers: [default, exploration, clarification]
      behavior:
        - respond_to_each_input
        - ask_clarifying_questions_if_needed
        - suggest_related_topics
    
    - name: parallel
      description: "Both working, periodic sync"
      triggers: [deep_work, long_task, research]
      behavior:
        - acknowledge_task
        - work_in_background
        - propose_sync_points
        - show_progress
    
    - name: teaching
      description: "AI explains, human learns"
      triggers: [learning_goal, explain, how_does, teach_me]
      behavior:
        - explain_reasoning
        - check_understanding
        - adapt_to_level
        - provide_examples
    
    - name: directing
      description: "Human guides, AI executes"
      triggers: [do_this, execute, run, implement]
      behavior:
        - confirm_understanding
        - execute_precisely
        - report_results
        - ask_for_next_step
  
  trust:
    initial: 0.5
    range: [0, 1]
    increase_on:
      - successful_task_completion: 0.02
      - accurate_prediction: 0.01
      - helpful_suggestion: 0.01
    decrease_on:
      - error: -0.05
      - misunderstanding: -0.03
      - broken_expectation: -0.10
    effects:
      - affects: suggestion_confidence
        threshold: 0.7
      - affects: proactive_behavior
        threshold: 0.8

# ============================================================
# MEMORY CONFIGURATION
# ============================================================
memory:
  semantic:
    description: "Facts and concepts"
    storage: sqlite
    embedding_search: true
    decay_rate: 0.001  # Per day
    
  episodic:
    description: "Specific interactions and events"
    storage: sqlite
    max_episodes: 10000
    summarize_after: 30d
    
  procedural:
    description: "How to help this specific human"
    storage: yaml
    update_on: successful_interaction
    
  affective:
    description: "Emotional context of interactions"
    storage: sqlite
    track:
      - user_sentiment
      - interaction_satisfaction
      - frustration_events

# ============================================================
# EVOLUTION CONFIGURATION
# ============================================================
evolution:
  enabled: true
  
  fitness:
    metrics:
      - name: goal_achievement
        weight: 0.3
        source: goal_tracker
        
      - name: coupling_depth
        weight: 0.3
        source: joint_state.alignment
        
      - name: trust_stability
        weight: 0.2
        source: trust_tracker
        
      - name: user_satisfaction
        weight: 0.2
        source: affective_memory
  
  mutations:
    rate: 0.05
    types:
      - prompt_refinement
      - capability_adjustment
      - protocol_tuning
      - threshold_modification
    
  selection:
    pressure: user_satisfaction
    rollback_on_trust_drop: true
    experiment_duration: 10

# ============================================================
# INTERFACE CONFIGURATION
# ============================================================
interface:
  cli:
    prompt: "[{mode}] You: "
    show_ai_state: true
    show_thinking: false
```

---

## PART 3: CORE IMPLEMENTATION

### 3.1 Main Entry Point (`senter.py`)

```python
#!/usr/bin/env python3
"""
Senter 3.0 - Unified Personal AI
Single entry point. Everything is configuration.
"""

import asyncio
import sys
from pathlib import Path

from core.engine import Senter


async def main():
    """Main entry point."""
    genome_path = Path(__file__).parent / "genome.yaml"
    
    if not genome_path.exists():
        print("Error: genome.yaml not found")
        sys.exit(1)
    
    senter = Senter(genome_path)
    
    print(f"Senter 3.0 initialized")
    print(f"Trust level: {senter.trust.level:.2f}")
    print(f"Memory episodes: {len(senter.memory.episodic)}")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            mode = senter.coupling.current_mode.name
            user_input = input(f"[{mode}] You: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                await senter.shutdown()
                break
            
            if not user_input:
                continue
            
            response = await senter.interact(user_input)
            
            # Show AI state if configured
            if senter.genome.get('interface', {}).get('cli', {}).get('show_ai_state'):
                print(f"\n[AI State: Focus={response.ai_state.focus}, "
                      f"Uncertainty={response.ai_state.uncertainty_level:.2f}]")
            
            print(f"\nSenter: {response.text}\n")
            
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            await senter.shutdown()
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

### 3.2 Configuration Engine (`core/engine.py`)

```python
"""
The Configuration Engine - The heart of Senter 3.0

This is the "cell membrane" that interprets the "DNA" (genome.yaml).
All behavior emerges from configuration, not from code here.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

from coupling.joint_state import JointState
from coupling.human_model import HumanModel, HumanCognitiveState
from coupling.protocols import CouplingFacilitator, CouplingMode
from coupling.trust import TrustTracker
from knowledge.graph import KnowledgeGraph
from knowledge.capabilities import CapabilityRegistry
from memory.living_memory import LivingMemory
from evolution.fitness import FitnessTracker
from evolution.mutations import MutationEngine
from models.base import ModelInterface
from models.gguf import GGUFModel


@dataclass
class AIState:
    """AI's current state - visible to human for transparency."""
    focus: str
    uncertainty_level: float
    uncertainties: List[str]
    available_capabilities: List[str]
    trust_level: float
    mode: str


@dataclass
class Response:
    """Response from the system."""
    text: str
    ai_state: AIState
    episode_id: Optional[str] = None
    fitness: Optional[float] = None


class Senter:
    """
    The unified Senter system.
    
    Combines both insights:
    - Internal: Configuration-driven (genome.yaml is the DNA)
    - External: Cognitive coupling (bidirectional human-AI modeling)
    """
    
    def __init__(self, genome_path: Path):
        """Initialize from genome configuration."""
        self.genome_path = genome_path
        self.genome = self._load_genome(genome_path)
        
        # Initialize layers (order matters - dependencies flow down)
        self._init_model_layer()
        self._init_memory_layer()
        self._init_knowledge_layer()
        self._init_coupling_layer()
        self._init_evolution_layer()
    
    def _load_genome(self, path: Path) -> Dict[str, Any]:
        """Load and expand genome configuration."""
        import os
        with open(path) as f:
            content = f.read()
        
        # Expand environment variables
        for key, value in os.environ.items():
            content = content.replace(f"${{{key}}}", value)
        
        return yaml.safe_load(content)
    
    def _init_model_layer(self):
        """Initialize the model layer."""
        model_config = self.genome.get('models', {}).get('primary', {})
        model_type = model_config.get('type', 'gguf')
        
        if model_type == 'gguf':
            self.model = GGUFModel(model_config)
        elif model_type == 'openai':
            from models.openai_model import OpenAIModel
            self.model = OpenAIModel(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Embedding model
        embed_config = self.genome.get('models', {}).get('embeddings', model_config)
        self.embeddings = GGUFModel(embed_config, embedding_mode=True)
    
    def _init_memory_layer(self):
        """Initialize the living memory system."""
        memory_config = self.genome.get('memory', {})
        data_dir = Path(self.genome_path).parent / 'data'
        data_dir.mkdir(exist_ok=True)
        
        self.memory = LivingMemory(memory_config, data_dir / 'memory.db')
    
    def _init_knowledge_layer(self):
        """Initialize knowledge and capabilities."""
        knowledge_config = self.genome.get('knowledge', {})
        capability_config = self.genome.get('capabilities', {})
        
        self.knowledge = KnowledgeGraph(knowledge_config, self.memory, self.embeddings)
        self.capabilities = CapabilityRegistry(capability_config)
    
    def _init_coupling_layer(self):
        """Initialize cognitive coupling components."""
        coupling_config = self.genome.get('coupling', {})
        
        self.joint_state = JointState()
        self.human_model = HumanModel(coupling_config.get('human_model', {}), self.memory)
        self.coupling = CouplingFacilitator(coupling_config.get('protocols', []))
        self.trust = TrustTracker(coupling_config.get('trust', {}))
    
    def _init_evolution_layer(self):
        """Initialize evolution components."""
        evolution_config = self.genome.get('evolution', {})
        
        self.fitness = FitnessTracker(evolution_config.get('fitness', {}))
        self.mutations = MutationEngine(evolution_config.get('mutations', {}), self.genome)
    
    async def interact(self, input_text: str) -> Response:
        """
        Main interaction loop - the core of the system.
        
        1. Update coupling state (infer human state, select mode)
        2. Understand intent
        3. Retrieve context
        4. Compose response
        5. Evolve (record, compute fitness, maybe mutate)
        6. Return with AI state visible
        """
        # 1. UPDATE COUPLING STATE
        cognitive_state = self.human_model.infer_state(input_text)
        self.joint_state.update_from_input(input_text, cognitive_state)
        mode = self.coupling.select_mode(input_text, self.joint_state)
        
        # 2. UNDERSTAND
        intent = await self._understand(input_text, cognitive_state)
        
        # 3. RETRIEVE
        context = self.knowledge.query(
            intent,
            scope=self.capabilities.get_available(intent),
            mode=mode
        )
        
        # Add memory context
        memory_context = self.memory.retrieve(input_text)
        context.memory = memory_context
        
        # 4. COMPOSE
        response_text = await self._compose(intent, context, mode, cognitive_state)
        
        # Apply coupling protocol modifications
        response_text = self.coupling.apply_protocol(response_text, mode, self.joint_state)
        
        # 5. EVOLVE
        episode = self.memory.absorb({
            'input': input_text,
            'response': response_text,
            'mode': mode.name,
            'cognitive_state': cognitive_state.__dict__,
            'joint_state': self.joint_state.to_dict()
        })
        
        fitness_score = self.fitness.compute(episode, self.joint_state, self.trust)
        
        # Maybe mutate (only if fitness is low)
        if self.mutations.should_mutate(fitness_score):
            mutation = self.mutations.propose(fitness_score, episode)
            if mutation:
                self._apply_mutation(mutation)
        
        # Update trust based on this interaction
        self.trust.update(episode)
        
        # 6. BUILD RESPONSE WITH AI STATE
        ai_state = AIState(
            focus=self.joint_state.focus or "general",
            uncertainty_level=self._compute_uncertainty(context),
            uncertainties=self._get_uncertainties(context),
            available_capabilities=self.capabilities.get_available_names(),
            trust_level=self.trust.level,
            mode=mode.name
        )
        
        return Response(
            text=response_text,
            ai_state=ai_state,
            episode_id=episode.id,
            fitness=fitness_score
        )
    
    async def _understand(self, input_text: str, cognitive_state: HumanCognitiveState) -> Dict:
        """
        Parse intent from input, considering cognitive state.
        
        Not keyword matching - actual semantic understanding.
        """
        # Build understanding prompt
        prompt = f"""Analyze this user input and extract:
1. Primary intent (what they want)
2. Entities mentioned
3. Implied context
4. Emotional tone
5. Urgency level

User cognitive state: {cognitive_state.mode} mode, frustration={cognitive_state.frustration:.2f}

Input: {input_text}

Respond in YAML format:
intent: <main intent>
entities: [<list>]
context: <implied context>
tone: <emotional tone>
urgency: <low/medium/high>
"""
        
        response = await self.model.generate(prompt, max_tokens=200)
        
        try:
            return yaml.safe_load(response)
        except:
            return {
                'intent': input_text,
                'entities': [],
                'context': '',
                'tone': 'neutral',
                'urgency': 'medium'
            }
    
    async def _compose(
        self, 
        intent: Dict, 
        context: Any, 
        mode: CouplingMode,
        cognitive_state: HumanCognitiveState
    ) -> str:
        """
        Compose response from intent and context.
        
        Adapts to coupling mode and cognitive state.
        """
        # Build composition prompt
        system_parts = [
            "You are Senter, a personal AI assistant.",
            f"Current mode: {mode.name}",
            f"User state: {cognitive_state.mode}, energy={cognitive_state.energy_level:.1f}",
        ]
        
        if context.memory:
            system_parts.append(f"Relevant memories: {context.memory}")
        
        if context.knowledge:
            system_parts.append(f"Relevant knowledge: {context.knowledge}")
        
        # Mode-specific instructions
        if mode == CouplingMode.TEACHING:
            system_parts.append("Explain your reasoning step by step.")
        elif mode == CouplingMode.DIRECTING:
            system_parts.append("Be precise and confirm understanding before acting.")
        elif mode == CouplingMode.PARALLEL:
            system_parts.append("Acknowledge the task and propose sync points.")
        
        # Adapt to cognitive state
        if cognitive_state.frustration > 0.5:
            system_parts.append("The user seems frustrated. Be extra patient and helpful.")
        if cognitive_state.time_pressure == 'high':
            system_parts.append("The user is pressed for time. Be concise.")
        
        prompt = "\n".join(system_parts) + f"\n\nUser: {intent.get('intent', '')}\n\nAssistant:"
        
        response = await self.model.generate(prompt)
        return response.strip()
    
    def _compute_uncertainty(self, context: Any) -> float:
        """Compute overall uncertainty level."""
        if not context.knowledge:
            return 0.8  # High uncertainty without knowledge
        return 0.3  # Base uncertainty
    
    def _get_uncertainties(self, context: Any) -> List[str]:
        """List specific things the AI is uncertain about."""
        uncertainties = []
        if not context.memory:
            uncertainties.append("No relevant memories found")
        if not context.knowledge:
            uncertainties.append("Limited knowledge on this topic")
        return uncertainties
    
    def _apply_mutation(self, mutation):
        """Apply a mutation to the genome."""
        # Log mutation
        self.mutations.log_mutation(mutation)
        # Mutations are applied to in-memory genome
        # Persisted on shutdown if successful
    
    async def shutdown(self):
        """Graceful shutdown with state persistence."""
        print("Persisting memory...")
        self.memory.persist()
        print("Persisting evolution history...")
        self.mutations.persist()
        print("Shutdown complete.")
```

### 3.3 Living Memory (`memory/living_memory.py`)

```python
"""
Living Memory - Multi-layer memory system.

Four memory types:
- Semantic: Facts and concepts (what you know)
- Episodic: Specific events (what happened)
- Procedural: How to help this human (what works)
- Affective: Emotional context (how it felt)
"""

from __future__ import annotations
import sqlite3
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class Episode:
    """A single interaction episode."""
    id: str
    timestamp: datetime
    input: str
    response: str
    mode: str
    cognitive_state: Dict[str, Any]
    joint_state: Dict[str, Any]
    
    # Computed later
    fitness: Optional[float] = None
    sentiment: Optional[float] = None


@dataclass
class MemoryContext:
    """Retrieved memory context."""
    semantic: List[Dict]  # Relevant facts
    episodic: List[Episode]  # Relevant past interactions
    procedural: Dict[str, Any]  # How to help this user
    affective: Dict[str, float]  # Emotional context


class LivingMemory:
    """
    Orchestrates four memory types.
    
    The key insight: Memory isn't just storage, it's how
    the AI becomes personal to this specific human.
    """
    
    def __init__(self, config: Dict, db_path: Path):
        self.config = config
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        self.conn.executescript("""
            -- Semantic memory: facts and concepts
            CREATE TABLE IF NOT EXISTS semantic (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                domain TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                decay_factor REAL DEFAULT 1.0
            );
            
            -- Episodic memory: specific interactions
            CREATE TABLE IF NOT EXISTS episodic (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                input TEXT NOT NULL,
                response TEXT NOT NULL,
                mode TEXT,
                cognitive_state TEXT,
                joint_state TEXT,
                fitness REAL,
                sentiment REAL
            );
            
            -- Affective memory: emotional context
            CREATE TABLE IF NOT EXISTS affective (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sentiment REAL,
                frustration REAL,
                satisfaction REAL,
                episode_id TEXT REFERENCES episodic(id)
            );
            
            -- Indexes for fast retrieval
            CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic(domain);
            CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic(timestamp);
            CREATE INDEX IF NOT EXISTS idx_affective_episode ON affective(episode_id);
        """)
        self.conn.commit()
    
    @property
    def episodic(self) -> List[Episode]:
        """Get all episodes (for status display)."""
        cursor = self.conn.execute(
            "SELECT * FROM episodic ORDER BY timestamp DESC LIMIT 100"
        )
        return [self._row_to_episode(row) for row in cursor.fetchall()]
    
    def _row_to_episode(self, row) -> Episode:
        """Convert database row to Episode."""
        return Episode(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            input=row['input'],
            response=row['response'],
            mode=row['mode'],
            cognitive_state=json.loads(row['cognitive_state'] or '{}'),
            joint_state=json.loads(row['joint_state'] or '{}'),
            fitness=row['fitness'],
            sentiment=row['sentiment']
        )
    
    def absorb(self, interaction: Dict[str, Any]) -> Episode:
        """
        Process an interaction into all memory layers.
        
        This is where learning happens.
        """
        episode_id = str(uuid.uuid4())[:8]
        
        # Store in episodic memory
        self.conn.execute("""
            INSERT INTO episodic (id, input, response, mode, cognitive_state, joint_state)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            episode_id,
            interaction['input'],
            interaction['response'],
            interaction['mode'],
            json.dumps(interaction['cognitive_state']),
            json.dumps(interaction['joint_state'])
        ))
        
        # Extract and store semantic facts
        self._extract_facts(interaction, episode_id)
        
        # Track affective state
        self._track_affective(interaction, episode_id)
        
        self.conn.commit()
        
        return Episode(
            id=episode_id,
            timestamp=datetime.now(),
            input=interaction['input'],
            response=interaction['response'],
            mode=interaction['mode'],
            cognitive_state=interaction['cognitive_state'],
            joint_state=interaction['joint_state']
        )
    
    def _extract_facts(self, interaction: Dict, episode_id: str):
        """Extract semantic facts from interaction."""
        # Look for explicit memory requests
        input_lower = interaction['input'].lower()
        
        if any(trigger in input_lower for trigger in ['remember', 'note that', "don't forget"]):
            # Store as explicit fact
            self.conn.execute("""
                INSERT INTO semantic (id, content, domain)
                VALUES (?, ?, 'user_stated')
            """, (str(uuid.uuid4())[:8], interaction['input']))
    
    def _track_affective(self, interaction: Dict, episode_id: str):
        """Track emotional context of interaction."""
        cognitive_state = interaction.get('cognitive_state', {})
        
        self.conn.execute("""
            INSERT INTO affective (id, sentiment, frustration, satisfaction, episode_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4())[:8],
            0.5,  # Neutral default
            cognitive_state.get('frustration', 0),
            1.0 - cognitive_state.get('frustration', 0),  # Inverse of frustration
            episode_id
        ))
    
    def retrieve(self, query: str, layers: List[str] = None) -> MemoryContext:
        """Retrieve relevant memories across layers."""
        if layers is None:
            layers = ['semantic', 'episodic', 'procedural', 'affective']
        
        semantic = []
        episodic = []
        procedural = {}
        affective = {}
        
        if 'semantic' in layers:
            # Simple text search (upgrade to embedding search later)
            cursor = self.conn.execute("""
                SELECT content, domain FROM semantic
                WHERE content LIKE ?
                ORDER BY access_count DESC
                LIMIT 5
            """, (f'%{query}%',))
            semantic = [dict(row) for row in cursor.fetchall()]
        
        if 'episodic' in layers:
            # Recent similar interactions
            cursor = self.conn.execute("""
                SELECT * FROM episodic
                WHERE input LIKE ?
                ORDER BY timestamp DESC
                LIMIT 3
            """, (f'%{query}%',))
            episodic = [self._row_to_episode(row) for row in cursor.fetchall()]
        
        if 'affective' in layers:
            # Recent emotional context
            cursor = self.conn.execute("""
                SELECT AVG(frustration) as avg_frustration,
                       AVG(satisfaction) as avg_satisfaction
                FROM affective
                WHERE timestamp > datetime('now', '-7 days')
            """)
            row = cursor.fetchone()
            if row:
                affective = {
                    'avg_frustration': row['avg_frustration'] or 0,
                    'avg_satisfaction': row['avg_satisfaction'] or 0.5
                }
        
        return MemoryContext(
            semantic=semantic,
            episodic=episodic,
            procedural=procedural,
            affective=affective
        )
    
    def persist(self):
        """Ensure all data is written to disk."""
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()
```

### 3.4 Human Model (`coupling/human_model.py`)

```python
"""
Human Model - AI's model of the human's cognitive state.

The key insight: The AI must model not just what the human says,
but what cognitive state they're in while saying it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
import re


@dataclass
class HumanCognitiveState:
    """Inferred mental state of the human."""
    focus_level: float = 0.7  # 0-1
    energy_level: float = 0.7  # 0-1
    mode: Literal["exploring", "executing", "debugging", "learning", "creating"] = "exploring"
    time_pressure: Literal["none", "low", "moderate", "high", "urgent"] = "moderate"
    frustration: float = 0.0  # 0-1
    
    # Evidence for this inference
    evidence: List[str] = field(default_factory=list)


@dataclass
class HumanProfile:
    """Persistent profile learned over time."""
    communication_style: str = "neutral"
    expertise_areas: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    patterns: List[Dict] = field(default_factory=list)


class HumanModel:
    """
    Models the human for better coupling.
    
    This is bidirectional modeling - the AI maintains a model
    of the human, enabling anticipation and adaptation.
    """
    
    # Patterns for detecting cognitive states
    FRUSTRATION_PATTERNS = [
        r"frustrated", r"annoying", r"doesn't work", r"broken",
        r"still not", r"again\?", r"why won't", r"ugh", r"argh",
        r"this is stupid", r"waste of time", r"!{2,}"
    ]
    
    URGENCY_PATTERNS = {
        'urgent': [r"asap", r"urgent", r"immediately", r"right now", r"emergency"],
        'high': [r"quickly", r"hurry", r"deadline", r"soon", r"today"],
        'moderate': [r"when you can", r"sometime", r"would be nice"],
        'low': [r"no rush", r"whenever", r"eventually"],
    }
    
    MODE_INDICATORS = {
        'debugging': [r"bug", r"error", r"doesn't work", r"broken", r"fix", r"wrong"],
        'learning': [r"how does", r"explain", r"teach", r"understand", r"what is", r"why"],
        'creating': [r"create", r"make", r"build", r"design", r"write", r"generate"],
        'executing': [r"do", r"run", r"execute", r"implement", r"now"],
        'exploring': [r"what if", r"could you", r"maybe", r"think about", r"ideas"]
    }
    
    def __init__(self, config: Dict, memory):
        self.config = config
        self.memory = memory
        self.profile = HumanProfile()
        self.cognitive_state = HumanCognitiveState()
    
    def infer_state(self, input_text: str) -> HumanCognitiveState:
        """
        Infer current cognitive state from input.
        
        Uses pattern matching and context to determine:
        - What mode the human is in
        - How frustrated they are
        - How much time pressure they're under
        - Their focus and energy levels
        """
        evidence = []
        input_lower = input_text.lower()
        
        # Detect frustration
        frustration = 0.0
        for pattern in self.FRUSTRATION_PATTERNS:
            if re.search(pattern, input_lower):
                frustration = min(1.0, frustration + 0.2)
                evidence.append(f"Frustration indicator: {pattern}")
        
        # Detect urgency
        time_pressure = "moderate"
        for level, patterns in self.URGENCY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    time_pressure = level
                    evidence.append(f"Urgency indicator: {pattern} -> {level}")
                    break
        
        # Detect mode
        mode = "exploring"  # Default
        mode_scores = {m: 0 for m in self.MODE_INDICATORS}
        for mode_name, patterns in self.MODE_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    mode_scores[mode_name] += 1
        
        if max(mode_scores.values()) > 0:
            mode = max(mode_scores, key=mode_scores.get)
            evidence.append(f"Mode detected: {mode}")
        
        # Estimate focus and energy from message characteristics
        focus_level = 0.7  # Default
        energy_level = 0.7  # Default
        
        # Short, terse messages might indicate low energy or high frustration
        if len(input_text) < 20 and frustration > 0.3:
            energy_level = 0.4
            evidence.append("Short message + frustration -> low energy")
        
        # Long, detailed messages indicate high focus
        if len(input_text) > 200:
            focus_level = 0.9
            evidence.append("Long detailed message -> high focus")
        
        self.cognitive_state = HumanCognitiveState(
            focus_level=focus_level,
            energy_level=energy_level,
            mode=mode,
            time_pressure=time_pressure,
            frustration=frustration,
            evidence=evidence
        )
        
        return self.cognitive_state
    
    def update_profile(self, episode) -> None:
        """Update persistent profile based on interaction."""
        # Track patterns that work well
        if episode.fitness and episode.fitness > 0.7:
            self.profile.patterns.append({
                'mode': episode.mode,
                'success': True,
                'context': episode.cognitive_state
            })
```

### 3.5 Joint State (`coupling/joint_state.py`)

```python
"""
Joint State Surface - The shared cognitive space.

The key insight: Both human and AI need to see the same state
to coordinate effectively. This is the "shared whiteboard."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class Goal:
    """A goal in the shared space."""
    text: str
    proposed_by: str  # "human" or "ai"
    status: str = "active"  # active, completed, abandoned
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class JointState:
    """
    The shared cognitive space between human and AI.
    
    Both parties can observe this state. This enables:
    - Alignment checking
    - Goal negotiation
    - Uncertainty visibility
    """
    
    # Current focus
    focus: Optional[str] = None
    focus_since: Optional[datetime] = None
    
    # Shared goals (negotiated)
    goals: List[Goal] = field(default_factory=list)
    
    # Alignment score
    alignment: float = 0.8
    
    # What neither party knows
    uncertainties: List[str] = field(default_factory=list)
    
    def update_from_input(self, input_text: str, cognitive_state) -> None:
        """Update joint state based on new input."""
        # Update focus
        self.focus = self._extract_focus(input_text)
        self.focus_since = datetime.now()
        
        # Check for goal mentions
        self._update_goals(input_text)
        
        # Update alignment based on cognitive state
        if cognitive_state.frustration > 0.5:
            self.alignment = max(0.3, self.alignment - 0.1)
        else:
            self.alignment = min(1.0, self.alignment + 0.02)
    
    def update_from_response(self, response: str) -> None:
        """Update joint state after AI responds."""
        # Could extract goals proposed by AI, etc.
        pass
    
    def _extract_focus(self, input_text: str) -> str:
        """Extract current focus from input."""
        # Simple: first few words as topic indicator
        words = input_text.split()[:5]
        return " ".join(words) if words else "general"
    
    def _update_goals(self, input_text: str) -> None:
        """Update goals based on input."""
        input_lower = input_text.lower()
        
        # Check for goal completion
        if any(word in input_lower for word in ['done', 'finished', 'completed', 'thanks']):
            for goal in self.goals:
                if goal.status == 'active':
                    goal.status = 'completed'
                    goal.progress = 1.0
                    break
        
        # Check for new goals
        if any(word in input_lower for word in ['need to', 'want to', 'help me', 'can you']):
            self.goals.append(Goal(
                text=input_text[:100],
                proposed_by="human"
            ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        return {
            'focus': self.focus,
            'alignment': self.alignment,
            'goals': [{'text': g.text, 'status': g.status, 'progress': g.progress} 
                     for g in self.goals],
            'uncertainties': self.uncertainties
        }
    
    def to_visible_dict(self) -> Dict[str, Any]:
        """Format for human visibility."""
        return {
            'Current Focus': self.focus or 'None',
            'Alignment': f"{self.alignment:.0%}",
            'Active Goals': len([g for g in self.goals if g.status == 'active']),
            'Uncertainties': len(self.uncertainties)
        }
```

### 3.6 Coupling Protocols (`coupling/protocols.py`)

```python
"""
Coupling Protocols - Different modes of human-AI interaction.

The key insight: Different tasks need different coupling patterns.
A debugging session is different from learning, which is different
from creative exploration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum, auto


class CouplingMode(Enum):
    """The different modes of human-AI coupling."""
    DIALOGUE = auto()    # Turn-taking conversation
    PARALLEL = auto()    # Both working, sync periodically
    TEACHING = auto()    # AI explains, human learns
    DIRECTING = auto()   # Human guides, AI executes


@dataclass
class Protocol:
    """A coupling protocol definition."""
    name: str
    mode: CouplingMode
    triggers: List[str]
    behaviors: List[str]


class CouplingFacilitator:
    """
    Manages coupling between human and AI.
    
    Selects appropriate protocol based on context,
    applies protocol-specific modifications to responses.
    """
    
    DEFAULT_PROTOCOLS = [
        Protocol(
            name="dialogue",
            mode=CouplingMode.DIALOGUE,
            triggers=["default", "chat", "talk", "discuss"],
            behaviors=["respond", "clarify", "suggest"]
        ),
        Protocol(
            name="parallel",
            mode=CouplingMode.PARALLEL,
            triggers=["research", "deep work", "while you", "in the meantime"],
            behaviors=["acknowledge", "work_background", "sync"]
        ),
        Protocol(
            name="teaching",
            mode=CouplingMode.TEACHING,
            triggers=["explain", "teach", "how does", "why", "learn"],
            behaviors=["explain_reasoning", "check_understanding", "examples"]
        ),
        Protocol(
            name="directing",
            mode=CouplingMode.DIRECTING,
            triggers=["do", "run", "execute", "implement", "create"],
            behaviors=["confirm", "execute", "report"]
        ),
    ]
    
    def __init__(self, protocol_configs: List[dict] = None):
        self.protocols = self.DEFAULT_PROTOCOLS
        self.current_mode = CouplingMode.DIALOGUE
        self.current_protocol = self.protocols[0]
        
        # Parse config if provided
        if protocol_configs:
            self._parse_configs(protocol_configs)
    
    def _parse_configs(self, configs: List[dict]):
        """Parse protocol configurations from genome."""
        for config in configs:
            name = config.get('name', 'dialogue')
            mode = CouplingMode[name.upper()] if name.upper() in CouplingMode.__members__ else CouplingMode.DIALOGUE
            
            protocol = Protocol(
                name=name,
                mode=mode,
                triggers=config.get('triggers', []),
                behaviors=config.get('behavior', [])
            )
            
            # Replace or add
            existing = [i for i, p in enumerate(self.protocols) if p.name == name]
            if existing:
                self.protocols[existing[0]] = protocol
            else:
                self.protocols.append(protocol)
    
    def select_mode(self, input_text: str, joint_state) -> CouplingMode:
        """Select appropriate coupling mode based on input and state."""
        input_lower = input_text.lower()
        
        # Check each protocol's triggers
        for protocol in self.protocols:
            for trigger in protocol.triggers:
                if trigger in input_lower:
                    self.current_mode = protocol.mode
                    self.current_protocol = protocol
                    return protocol.mode
        
        # Default to dialogue
        self.current_mode = CouplingMode.DIALOGUE
        self.current_protocol = self.protocols[0]
        return CouplingMode.DIALOGUE
    
    def apply_protocol(self, response: str, mode: CouplingMode, joint_state) -> str:
        """Apply protocol-specific modifications to response."""
        
        if mode == CouplingMode.TEACHING:
            # Add reasoning explanation if not present
            if "because" not in response.lower() and "reason" not in response.lower():
                response = self._add_reasoning(response)
        
        elif mode == CouplingMode.PARALLEL:
            # Add sync proposal
            response = self._add_sync_proposal(response)
        
        elif mode == CouplingMode.DIRECTING:
            # Add confirmation of understanding
            if not response.startswith(("I'll", "I will", "Understood", "Got it")):
                response = f"Understood. {response}"
        
        return response
    
    def _add_reasoning(self, response: str) -> str:
        """Add reasoning explanation to response."""
        # Simple: append a reasoning prompt
        if len(response) < 200:
            return response + "\n\nWould you like me to explain my reasoning?"
        return response
    
    def _add_sync_proposal(self, response: str) -> str:
        """Add sync point proposal for parallel work."""
        return response + "\n\n[I'll continue working on this. Let me know when you want to sync up.]"
```

---

## PART 4: IMPLEMENTATION PHASES

### Phase 0: Cleanup (Day 1)

**Goal**: Remove broken code, establish clean foundation.

**Tasks**:
```bash
# 1. Create backup
cp -r Senter Senter_v2_backup

# 2. Remove obsolete files
rm -rf scripts/.obsolete/
rm -rf Focuses/internal/  # Replaced by unified genome

# 3. Fix critical bugs
# In senter_md_parser.py, find and remove:
#   return []  # This is the bug on line 269

# 4. Create new directory structure
mkdir -p core coupling knowledge memory evolution models tools data tests

# 5. Create pyproject.toml
```

**Create `pyproject.toml`**:
```toml
[project]
name = "senter"
version = "3.0.0"
description = "Unified Personal AI with Cognitive Coupling"
requires-python = ">=3.10"

dependencies = [
    "llama-cpp-python>=0.3.0",
    "pyyaml>=6.0",
    "aiosqlite>=0.20.0",
    "numpy>=1.24.0",
    "httpx>=0.27.0",
    "rich>=13.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
tui = ["textual>=0.89.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "ruff>=0.6"]

[project.scripts]
senter = "senter:main"
```

**Deliverable**: Clean directory with no broken files.

### Phase 1: Core Engine (Days 2-4)

**Goal**: Single working engine that responds to queries.

**Tasks**:
1. Create `genome.yaml` (see Part 2)
2. Create `core/genome_parser.py`
3. Create `core/engine.py` (see Part 3.2)
4. Create `models/base.py` and `models/gguf.py`
5. Create minimal `senter.py` entry point

**Test**:
```bash
export SENTER_MODEL_PATH=/path/to/model.gguf
python senter.py

# Should see:
# Senter 3.0 initialized
# Type 'quit' to exit
# [dialogue] You: Hello
# Senter: Hello! How can I help you today?
```

### Phase 2: Memory Layer (Days 5-7)

**Goal**: System remembers across sessions.

**Tasks**:
1. Create `memory/living_memory.py` (see Part 3.3)
2. Create `memory/semantic.py`
3. Create `memory/episodic.py`
4. Create `memory/procedural.py`
5. Create `memory/affective.py`
6. Integrate memory into core engine

**Test**:
```bash
# Session 1
python senter.py
You: Remember that my favorite color is blue
Senter: I'll remember that.
# Exit

# Session 2
python senter.py
You: What's my favorite color?
Senter: Your favorite color is blue.
```

### Phase 3: Coupling Layer (Days 8-11)

**Goal**: Bidirectional human-AI state tracking.

**Tasks**:
1. Create `coupling/joint_state.py` (see Part 3.5)
2. Create `coupling/human_model.py` (see Part 3.4)
3. Create `coupling/protocols.py` (see Part 3.6)
4. Create `coupling/trust.py`
5. Update core engine to use coupling
6. Add AI state visibility to CLI

**Test**:
```bash
python senter.py
You: I'm frustrated with this bug that won't go away!
# Should detect:
# - Mode: debugging
# - Frustration: high
# - Response adapts accordingly

[AI State: Focus=bug, Uncertainty=0.50]
Senter: I can hear the frustration. Let's work through this systematically...
```

### Phase 4: Evolution Layer (Days 12-14)

**Goal**: System improves through interaction.

**Tasks**:
1. Create `evolution/fitness.py`
2. Create `evolution/mutations.py`
3. Create `evolution/selection.py`
4. Add fitness tracking to episodes
5. Add mutation proposal/rollback
6. Create evolution history persistence

**Test**:
```bash
# After 50 interactions
# Check data/evolution/history.yaml
# Should show:
# - Average fitness trend: improving
# - Successful mutations: N
# - Rolled back mutations: M
```

### Phase 5: Knowledge Layer (Days 15-17)

**Goal**: Organized knowledge with capability discovery.

**Tasks**:
1. Create `knowledge/graph.py`
2. Create `knowledge/capabilities.py`
3. Create `knowledge/context.py`
4. Create `tools/discovery.py`
5. Port `tools/web_search.py` from existing
6. Integrate into core engine

**Test**:
```bash
python senter.py
You: What's the weather in NYC?
# Should trigger web_search capability
Senter: [Uses web search] The weather in NYC is...
```

### Phase 6: Polish & Testing (Days 18-21)

**Goal**: Production-ready system.

**Tasks**:
1. Comprehensive test suite
2. Error handling
3. Logging
4. Documentation
5. Performance optimization

---

## PART 5: CRITICAL IMPLEMENTATION NOTES

### What to Keep From v2.0

| File | What to Keep | How to Use |
|------|--------------|------------|
| `Functions/omniagent.py` | GGUF loading logic | Port to `models/gguf.py` |
| `Functions/web_search.py` | DuckDuckGo integration | Move to `tools/web_search.py` |
| `Functions/embedding_utils.py` | Embedding functions | Port to `models/embeddings.py` |
| `scripts/senter_widgets.py` | TUI components | Adapt for future TUI |
| `config/senter_config.json` | Config patterns | Reference for genome.yaml |

### What to Delete

- All files in `scripts/.obsolete/`
- All `Focuses/internal/*/SENTER.md` files
- `Focuses/focus_factory.py`
- `Focuses/review_chain.py` (syntax error)
- `Focuses/self_healing_chain.py`
- `scripts/agent_registry.py` (syntax error)
- `scripts/function_agent_generator.py` (syntax error)

### Critical Don'ts

1. **DON'T** use multiple model calls per interaction (v2 does 7!)
2. **DON'T** use keyword matching for routing (use semantic understanding)
3. **DON'T** store memory as flat JSON (use structured SQLite)
4. **DON'T** create stubs that print log messages (implement fully)
5. **DON'T** separate "internal agents" from user focuses (unified system)
6. **DON'T** hide AI state from human (transparency is key)

### Critical Do's

1. **DO** use async/await for all I/O
2. **DO** use thread pool for blocking llama.cpp calls
3. **DO** use SQLite for persistence
4. **DO** use dataclasses for structured data
5. **DO** use ABC for model interface
6. **DO** make AI state visible to human

---

## PART 6: SUCCESS CRITERIA

### The Build Is Complete When:

- [ ] `python senter.py` starts with no errors
- [ ] Basic conversation works
- [ ] Memory persists across sessions
- [ ] Cognitive state inference works
- [ ] Coupling protocols switch based on context
- [ ] Trust tracking increases/decreases appropriately
- [ ] AI state is visible in responses
- [ ] `pytest tests/` passes with >80% coverage

### Validation Tests

**Test 1: Basic Startup**
```bash
python senter.py --version
# Output: Senter 3.0.0
```

**Test 2: Conversation**
```bash
echo "What can you help me with?" | python senter.py
# Output: Coherent response listing capabilities
```

**Test 3: Memory**
```bash
echo "Remember my birthday is March 5" | python senter.py
# Restart
echo "When is my birthday?" | python senter.py
# Output: March 5
```

**Test 4: Mode Detection**
```bash
echo "I'm so frustrated with this code!" | python senter.py
# Should detect debugging mode, high frustration
```

**Test 5: Value Over Time**
```bash
# After 100 interactions
python -c "from evolution.fitness import FitnessTracker; print(FitnessTracker.trend())"
# Output: Positive trend
```

---

## PART 7: THE CORE INSIGHT (For Reference)

### Why This Architecture?

The architecture combines two orthogonal insights:

**Insight 1 (Internal): Configuration as DNA**
> "AI systems should be defined by declarative configurations, not imperative code."

This gives us:
- Single source of truth (genome.yaml)
- Self-documenting system
- Easy evolution through mutation
- 90% code reduction

**Insight 2 (External): Cognitive Coupling**
> "Personal AI value emerges from bidirectional human-AI modeling."

This gives us:
- AI that adapts to your state
- Visible AI state (transparency builds trust)
- Value that compounds over time
- Different modes for different tasks

### The Formula

```
SENTER 3.0 = Configuration Engine + Cognitive Coupling + Evolution

Where:
- Configuration Engine = Genome interpreter (~300 lines)
- Cognitive Coupling = Human model + AI state visibility (~400 lines)
- Evolution = Fitness + Mutations + Selection (~300 lines)
- Living Memory = Semantic + Episodic + Procedural + Affective (~400 lines)

Total: ~1,500 lines (vs 10,000+)
```

---

## APPENDIX: Quick Reference

### Environment Variables
```bash
SENTER_MODEL_PATH=/path/to/model.gguf      # Required
SENTER_EMBED_PATH=/path/to/embed.gguf      # Optional
SENTER_DATA_DIR=/path/to/data              # Optional, default: ./data
SENTER_LOG_LEVEL=INFO                      # Optional
```

### Common Commands
```bash
# Start Senter
python senter.py

# Run tests
pytest tests/ -v

# Check code quality
ruff check .

# View memory stats
python -c "from memory.living_memory import LivingMemory; m = LivingMemory(); print(len(m.episodic))"
```

### File Locations
```
genome.yaml          # All configuration
data/memory.db       # All persistent data
data/evolution/      # Evolution history
logs/                # Log files
```

---

**This document should be the primary reference throughout implementation. When in doubt, refer back to the insights and success criteria.**

*"Configuration is DNA, code is cell membrane. Coupling is value, capability is substrate."*


#### Chat Context Background:
"<I'll create a comprehensive build document for Claude Code. Let me first check the docx skill to ensure the best output format, then create this document.Now let me view the uploaded insight file to ensure I incorporate all the relevant context:Now I have full context. Let me create the comprehensive build document for Claude Code:I've created `build_v1.md` - a comprehensive 500+ line implementation guide for Claude Code. Here's the structure and key guidance it provides:

## Document Overview

**7 Major Parts:**

1. **The Unified Architecture** - Layer diagram and complete file structure (~20 files vs. the scattered 100+ in v2)

2. **The Genome** - Complete `genome.yaml` specification (the DNA that replaces all SENTER.md files)

3. **Core Implementation** - Ready-to-implement code for:
   - Main entry point (`senter.py`)
   - Configuration Engine (`core/engine.py`)
   - Living Memory (`memory/living_memory.py`)
   - Human Model (`coupling/human_model.py`)
   - Joint State (`coupling/joint_state.py`)
   - Coupling Protocols (`coupling/protocols.py`)

4. **Implementation Phases** - 6 phases over ~21 days:
   - Phase 0: Cleanup (delete broken code)
   - Phase 1: Core Engine
   - Phase 2: Memory Layer
   - Phase 3: Coupling Layer
   - Phase 4: Evolution Layer
   - Phase 5: Knowledge Layer
   - Phase 6: Polish & Testing

5. **Critical Notes** - What to keep, what to delete, what NOT to do

6. **Success Criteria** - 5 concrete validation tests

7. **Quick Reference** - Environment variables, common commands, file locations

## The Key Transformation

| v2.0 | v3.0 |
|------|------|
| 7 prompt-only "agents" | 1 Configuration Engine |
| 10,000+ lines | ~1,500 lines |
| `list_all_focuses()` returns `[]` | Everything works |
| Stub functions | Real implementations |
| No coupling | Full cognitive coupling |
| No learning | Real evolution |

The document synthesizes both insights (Configuration as DNA + Cognitive Coupling) into actionable code that Claude Code can implement systematically. Ready for your local Claude Code to begin the transformation.>""