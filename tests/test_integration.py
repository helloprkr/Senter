"""
Integration tests for Senter 3.0.
"""

import pytest
import yaml


class TestSenterIntegration:
    """Integration tests for the full system."""

    @pytest.fixture
    def genome_path(self, tmp_path):
        """Create a test genome file."""
        genome_content = {
            "version": "3.0",
            "name": "TestSenter",
            "models": {
                "primary": {
                    "type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                },
            },
            "coupling": {
                "trust": {"initial": 0.5},
                "protocols": [],
            },
            "memory": {},
            "evolution": {
                "enabled": True,
                "fitness": {},
                "mutations": {"rate": 0.0},
            },
            "knowledge": {},
            "capabilities": {
                "builtin": [
                    {"name": "respond", "always_available": True},
                ],
            },
            "interface": {
                "cli": {"show_ai_state": True},
            },
        }

        genome_file = tmp_path / "genome.yaml"
        with open(genome_file, "w") as f:
            yaml.dump(genome_content, f)

        return genome_file

    def test_genome_loading(self, genome_path):
        """Test that genome loads correctly."""
        from core.genome_parser import load_genome

        genome = load_genome(genome_path)

        assert genome.version == "3.0"
        assert genome.name == "TestSenter"

    def test_memory_initialization(self, genome_path):
        """Test memory layer initialization."""
        from memory.living_memory import LivingMemory

        db_path = genome_path.parent / "data" / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        memory = LivingMemory({}, db_path)

        assert memory is not None
        stats = memory.get_stats()
        assert stats["episodes"] == 0

    def test_coupling_initialization(self):
        """Test coupling layer initialization."""
        from coupling.joint_state import JointState
        from coupling.human_model import HumanModel
        from coupling.protocols import CouplingFacilitator
        from coupling.trust import TrustTracker

        joint_state = JointState()
        human_model = HumanModel({})
        coupling = CouplingFacilitator()
        trust = TrustTracker({"initial": 0.5})

        assert joint_state.alignment == 0.8
        assert trust.level == 0.5
        assert coupling.current_mode.name == "DIALOGUE"

    def test_evolution_initialization(self):
        """Test evolution layer initialization."""
        from evolution.fitness import FitnessTracker
        from evolution.mutations import MutationEngine
        from evolution.selection import SelectionPressure

        fitness = FitnessTracker()
        mutations = MutationEngine({})
        selection = SelectionPressure({})

        assert fitness is not None
        assert mutations is not None
        assert selection is not None

    def test_knowledge_initialization(self):
        """Test knowledge layer initialization."""
        from knowledge.graph import KnowledgeGraph
        from knowledge.capabilities import CapabilityRegistry
        from knowledge.context import ContextEngine

        knowledge = KnowledgeGraph()
        capabilities = CapabilityRegistry({
            "builtin": [
                {"name": "respond", "always_available": True},
            ],
        })
        context = ContextEngine()

        assert knowledge is not None
        assert "respond" in capabilities.get_available_names()
        assert context.turn_count == 0

    def test_full_interaction_flow(self, genome_path, tmp_path):
        """Test a complete interaction without actual LLM."""
        from memory.living_memory import LivingMemory
        from coupling.joint_state import JointState
        from coupling.human_model import HumanModel
        from coupling.protocols import CouplingFacilitator
        from coupling.trust import TrustTracker
        from evolution.fitness import FitnessTracker

        # Initialize components
        db_path = tmp_path / "test_memory.db"
        memory = LivingMemory({}, db_path)
        joint_state = JointState()
        human_model = HumanModel({})
        coupling = CouplingFacilitator()
        trust = TrustTracker({"initial": 0.5})
        fitness = FitnessTracker()

        # Simulate interaction
        input_text = "Hello, can you help me with something?"

        # 1. Infer cognitive state
        cognitive_state = human_model.infer_state(input_text)
        assert cognitive_state.mode in ("exploring", "executing", "debugging", "learning", "creating")

        # 2. Update joint state
        joint_state.update_from_input(input_text, cognitive_state)
        assert joint_state.focus is not None

        # 3. Select mode
        mode = coupling.select_mode(input_text, joint_state)
        assert mode is not None

        # 4. Simulate response and absorb
        response = "I'd be happy to help! What do you need?"
        episode = memory.absorb({
            "input": input_text,
            "response": response,
            "mode": mode.name,
            "cognitive_state": cognitive_state.to_dict(),
            "joint_state": joint_state.to_dict(),
        })

        assert episode.id is not None

        # 5. Compute fitness
        fitness_score = fitness.compute(episode, joint_state, trust)
        assert 0 <= fitness_score <= 1

        # 6. Update trust
        trust.update(episode)
        # Trust should stay roughly the same for neutral interaction
        assert 0.4 <= trust.level <= 0.6


class TestCLICommands:
    """Tests for CLI command handling."""

    def test_version_command(self):
        """Test version display."""
        # Would test the actual CLI command
        pass

    def test_help_command(self):
        """Test help display."""
        # Would test the actual CLI command
        pass


class TestWebSearch:
    """Tests for web search tool."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test that search returns results."""
        from tools.web_search import WebSearch

        searcher = WebSearch()
        try:
            results = await searcher.search("Python programming", max_results=3)
            # Results depend on API availability
            assert isinstance(results, list)
        finally:
            await searcher.close()


class TestFileOps:
    """Tests for file operations tool."""

    def test_read_write(self, tmp_path):
        """Test file read/write operations."""
        from tools.file_ops import FileOps

        ops = FileOps(allowed_paths=[tmp_path])

        # Write
        test_file = tmp_path / "test.txt"
        result = ops.write(test_file, "Hello, World!")
        assert result is True

        # Read
        content = ops.read(test_file)
        assert content == "Hello, World!"

    def test_path_validation(self, tmp_path):
        """Test path validation prevents escape."""
        from tools.file_ops import FileOps

        ops = FileOps(allowed_paths=[tmp_path])

        # Should not allow reading outside allowed paths
        content = ops.read("/etc/passwd")
        assert content is None

    def test_directory_operations(self, tmp_path):
        """Test directory operations."""
        from tools.file_ops import FileOps

        ops = FileOps(allowed_paths=[tmp_path])

        # Create directory
        new_dir = tmp_path / "subdir"
        result = ops.mkdir(new_dir)
        assert result is True
        assert ops.is_dir(new_dir)

        # List directory
        ops.write(new_dir / "file1.txt", "content")
        ops.write(new_dir / "file2.txt", "content")

        files = ops.list_dir(new_dir)
        assert len(files) == 2
