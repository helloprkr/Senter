"""
The Configuration Engine - The heart of Senter 3.0

This is the "cell membrane" that interprets the "DNA" (genome.yaml).
All behavior emerges from configuration, not from code here.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .genome_parser import GenomeParser
from .intent import IntentParser
from .composer import ResponseComposer, CompositionContext


@dataclass
class AIState:
    """AI's current state - visible to human for transparency."""

    focus: str
    uncertainty_level: float
    uncertainties: List[str]
    available_capabilities: List[str]
    trust_level: float
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "focus": self.focus,
            "uncertainty_level": self.uncertainty_level,
            "uncertainties": self.uncertainties,
            "available_capabilities": self.available_capabilities,
            "trust_level": self.trust_level,
            "mode": self.mode,
        }


@dataclass
class Response:
    """Response from the system."""

    text: str
    ai_state: AIState
    episode_id: Optional[str] = None
    fitness: Optional[float] = None
    suggestions: List[Dict[str, Any]] = None  # Proactive suggestions

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "ai_state": self.ai_state.to_dict(),
            "episode_id": self.episode_id,
            "fitness": self.fitness,
            "suggestions": self.suggestions,
        }


class Senter:
    """
    The unified Senter system.

    Combines both insights:
    - Internal: Configuration-driven (genome.yaml is the DNA)
    - External: Cognitive coupling (bidirectional human-AI modeling)
    """

    def __init__(self, genome_path: Path):
        """Initialize from genome configuration."""
        self.genome_path = Path(genome_path)
        self.genome = GenomeParser(genome_path)

        # Initialize layers (order matters - dependencies flow down)
        self._init_model_layer()
        self._init_memory_layer()
        self._init_knowledge_layer()
        self._init_coupling_layer()
        self._init_evolution_layer()
        self._init_intelligence_layer()

        # Core components
        self.intent_parser = IntentParser(self.model)
        self.composer = ResponseComposer(self.model, self.memory._procedural)

    def _init_model_layer(self) -> None:
        """Initialize the model layer."""
        from models.base import create_model

        model_config = self.genome.models.get("primary", {})

        try:
            self.model = create_model(model_config)
        except Exception as e:
            print(f"Warning: Could not initialize model: {e}")
            self.model = None

        # Embedding model
        embed_config = self.genome.models.get("embeddings", model_config)
        try:
            self.embeddings = create_model(embed_config)
        except Exception:
            self.embeddings = None

    def _init_memory_layer(self) -> None:
        """Initialize the living memory system."""
        from memory.living_memory import LivingMemory
        from models.embeddings import EmbeddingModel

        memory_config = self.genome.memory
        data_dir = self.genome_path.parent / "data"
        data_dir.mkdir(exist_ok=True)

        # Create embedding model wrapper for semantic search
        embedding_model = None
        if self.embeddings is not None:
            embedding_model = EmbeddingModel(self.embeddings)

        self.memory = LivingMemory(
            memory_config,
            data_dir / "memory.db",
            embeddings=embedding_model,
        )

    def _init_knowledge_layer(self) -> None:
        """Initialize knowledge and capabilities."""
        from knowledge.graph import KnowledgeGraph
        from knowledge.capabilities import CapabilityRegistry
        from knowledge.context import ContextEngine

        knowledge_config = self.genome.knowledge
        capability_config = self.genome.capabilities

        self.knowledge = KnowledgeGraph(knowledge_config, self.memory, self.embeddings)
        self.capabilities = CapabilityRegistry(capability_config)
        self.context = ContextEngine()

    def _init_coupling_layer(self) -> None:
        """Initialize cognitive coupling components."""
        import json
        from coupling.joint_state import JointState
        from coupling.human_model import HumanModel
        from coupling.protocols import CouplingFacilitator
        from coupling.trust import TrustTracker

        coupling_config = self.genome.coupling

        self.joint_state = JointState()
        self.human_model = HumanModel(coupling_config.get("human_model", {}), self.memory)
        self.coupling = CouplingFacilitator(coupling_config.get("protocols", []))
        self.trust = TrustTracker(coupling_config.get("trust", {}))

        # Load persisted trust level
        trust_file = self.genome_path.parent / "data" / "trust.json"
        if trust_file.exists():
            try:
                with open(trust_file) as f:
                    trust_data = json.load(f)
                    self.trust.level = trust_data.get("level", self.trust.level)
            except Exception:
                pass  # Use default if file is corrupted

    def _init_evolution_layer(self) -> None:
        """Initialize evolution components."""
        from evolution.fitness import FitnessTracker
        from evolution.mutations import MutationEngine
        from evolution.selection import SelectionPressure

        evolution_config = self.genome.evolution

        self.fitness = FitnessTracker(evolution_config.get("fitness", {}))
        self.mutations = MutationEngine(
            evolution_config.get("mutations", {}),
            self.genome.to_dict(),
            genome_path=self.genome_path,
        )
        self.selection = SelectionPressure(evolution_config.get("selection", {}))

        # Set persistence path for mutations
        data_dir = self.genome_path.parent / "data" / "evolution"
        self.mutations.set_persistence_path(data_dir)

    def _init_intelligence_layer(self) -> None:
        """Initialize intelligence components (goals, proactive suggestions)."""
        try:
            from intelligence.goals import GoalDetector
            from intelligence.proactive import ProactiveSuggestionEngine

            self.goal_detector = GoalDetector(self.memory)
            self.proactive = ProactiveSuggestionEngine(self, self.goal_detector)
        except Exception as e:
            print(f"Warning: Could not initialize intelligence layer: {e}")
            self.goal_detector = None
            self.proactive = None

    async def initialize(self) -> None:
        """Async initialization of models."""
        if self.model:
            await self.model.initialize()
        if self.embeddings and self.embeddings != self.model:
            await self.embeddings.initialize()

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
        intent = await self.intent_parser.parse(input_text, cognitive_state)

        # 3. RETRIEVE CONTEXT
        knowledge_result = self.knowledge.query(
            intent.to_dict(),
            scope=self.capabilities.get_available(intent.to_dict()),
            mode=mode,
        )

        # Add memory context
        memory_context = self.memory.retrieve(input_text)

        # ALWAYS retrieve user profile facts (name, preferences, etc.)
        user_profile = self._get_user_profile()

        # Get conversation history BEFORE adding current turn
        conversation_history = self.context.get_history_for_llm(limit=6)

        # Get active goals for context-aware responses
        active_goals = []
        if self.goal_detector:
            try:
                goals = self.goal_detector.get_active_goals()
                active_goals = [g.to_dict() for g in goals[:3]]  # Top 3 goals
            except Exception:
                pass  # Goal retrieval is optional

        context = CompositionContext(
            memory=memory_context,
            knowledge={"items": knowledge_result.knowledge} if knowledge_result.knowledge else None,
            capabilities=self.capabilities.get_available_names(),
            joint_state=self.joint_state.to_dict(),
            conversation_history=conversation_history,
            user_profile=user_profile,
            active_goals=active_goals,
        )

        # Update conversation context
        self.context.add_turn("user", input_text)

        # 4. COMPOSE RESPONSE
        response_text = await self.composer.compose(
            intent.to_dict(),
            context,
            mode,
            cognitive_state,
        )

        # Apply coupling protocol modifications
        response_text = self.coupling.apply_protocol(response_text, mode, self.joint_state)

        # Update joint state from response
        self.joint_state.update_from_response(response_text)

        # Add assistant turn to context
        self.context.add_turn("assistant", response_text)

        # 5. EVOLVE
        episode = self.memory.absorb(
            {
                "input": input_text,
                "response": response_text,
                "mode": mode.name,
                "cognitive_state": cognitive_state.to_dict(),
                "joint_state": self.joint_state.to_dict(),
            }
        )

        fitness_score = self.fitness.compute(episode, self.joint_state, self.trust)

        # Update episode with fitness
        self.memory.update_episode_fitness(episode.id, fitness_score)

        # Record interaction for active mutation experiment
        mutation_result = self.mutations.record_interaction(fitness_score)
        if mutation_result:
            # Experiment completed, update selection pressure
            self.selection.record_experiment_result(mutation_result)

        # Maybe mutate (only if fitness is low and no active experiment)
        if self.mutations.should_mutate(fitness_score):
            mutation = self.mutations.propose(fitness_score, episode)
            if mutation:
                # Start experiment to track mutation effectiveness
                self.selection.start_experiment(mutation, fitness_score)
                self.mutations.apply(mutation)

        # Update trust based on this interaction
        self.trust.update(episode)

        # Update human profile
        self.human_model.update_profile(episode)

        # Analyze for goals
        if self.goal_detector:
            try:
                self.goal_detector.analyze_interaction(input_text, response_text)
            except Exception:
                pass  # Goal detection is optional

        # Generate proactive suggestions (if trust is high enough)
        suggestions = []
        if self.proactive and self.trust.level >= 0.6:
            try:
                suggestions = await self.proactive.generate_suggestions()
            except Exception:
                pass  # Suggestions are optional

        # 6. BUILD RESPONSE WITH AI STATE
        ai_state = AIState(
            focus=self.joint_state.focus or "general",
            uncertainty_level=self._compute_uncertainty(context),
            uncertainties=self._get_uncertainties(context),
            available_capabilities=self.capabilities.get_available_names(),
            trust_level=self.trust.level,
            mode=mode.name,
        )

        return Response(
            text=response_text,
            ai_state=ai_state,
            episode_id=episode.id,
            fitness=fitness_score,
            suggestions=suggestions,
        )

    def _compute_uncertainty(self, context: CompositionContext) -> float:
        """Compute overall uncertainty level."""
        uncertainty = 0.3  # Base uncertainty

        if not context.memory or not context.memory.semantic:
            uncertainty += 0.2
        if not context.knowledge:
            uncertainty += 0.2

        return min(1.0, uncertainty)

    def _get_uncertainties(self, context: CompositionContext) -> List[str]:
        """List specific things the AI is uncertain about."""
        uncertainties = []

        if not context.memory or not context.memory.episodic:
            uncertainties.append("No relevant memories found")
        if not context.knowledge:
            uncertainties.append("Limited knowledge on this topic")

        return uncertainties

    def _get_user_profile(self) -> List[Dict[str, Any]]:
        """
        Get user profile facts that should ALWAYS be included in prompts.

        These are core facts about the user (name, preferences, etc.)
        that the AI should never forget.
        """
        profile_domains = ["user_name", "user_identity", "user_preference", "user_work", "user_role"]
        profile_facts = []

        for domain in profile_domains:
            facts = self.memory.semantic.get_by_domain(domain, limit=3)
            profile_facts.extend(facts)

        return profile_facts

    async def shutdown(self) -> None:
        """Graceful shutdown with state persistence."""
        import json

        print("Persisting memory...")
        self.memory.persist()

        print("Persisting trust level...")
        trust_file = self.genome_path.parent / "data" / "trust.json"
        with open(trust_file, "w") as f:
            json.dump({"level": self.trust.level, "events": len(self.trust._history)}, f)

        print("Persisting evolution history...")
        self.mutations.persist()

        if self.model:
            await self.model.close()
        if self.embeddings and self.embeddings != self.model:
            await self.embeddings.close()

        self.memory.close()
        print("Shutdown complete.")

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "version": self.genome.version,
            "name": self.genome.name,
            "memory_stats": self.memory.get_stats(),
            "trust": self.trust.to_dict(),
            "fitness": self.fitness.to_dict(),
            "coupling": self.coupling.to_dict(),
            "capabilities": self.capabilities.get_stats(),
        }
