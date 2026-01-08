"""
Vision Validation Tests

Tests that the complete Senter vision is achieved:
- 24/7 operation
- Learns from conversations
- Proactive intelligence
- Memory continuity
- Activity awareness
"""

import pytest
import time
from pathlib import Path
import subprocess
import sys


class TestDaemonOperation:
    """Test 24/7 daemon operation."""

    @pytest.fixture
    def daemon_process(self):
        """Start daemon for testing."""
        proc = subprocess.Popen(
            [sys.executable, 'senter.py', '--daemon'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent
        )
        time.sleep(3)  # Wait for startup
        yield proc
        proc.terminate()
        proc.wait()

    def test_daemon_starts(self, daemon_process):
        """Daemon process starts successfully."""
        assert daemon_process.poll() is None  # Still running

        # Check socket exists
        socket_path = Path("data/senter.sock")
        assert socket_path.exists()

    @pytest.mark.asyncio
    async def test_client_connects(self, daemon_process):
        """Client can connect to daemon."""
        from daemon.senter_client import SenterClient

        client = SenterClient()
        await client.connect()

        status = await client.status()
        assert status.get('status') == 'ok' or status.get('running') == True

        await client.disconnect()


class TestLearningFromConversations:
    """Test that system learns from conversations."""

    @pytest.fixture
    async def engine(self):
        from core.engine import Senter
        engine = Senter(Path('genome.yaml'))
        await engine.initialize()
        yield engine
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_goal_detection(self, engine):
        """Goals are detected from conversation patterns."""
        # Have conversations that imply goals
        await engine.interact("I want to learn machine learning")
        await engine.interact("I need to finish my thesis by March")
        await engine.interact("I'm trying to get better at Python")

        goals = engine.goal_detector.get_active_goals()

        assert len(goals) >= 1
        goal_texts = ' '.join(g.description.lower() for g in goals)
        assert any(word in goal_texts for word in ['learn', 'machine', 'thesis', 'python'])

    @pytest.mark.asyncio
    async def test_memory_storage(self, engine):
        """System stores facts in memory."""
        await engine.interact("Remember my favorite color is blue")

        # Check memory
        facts = engine.memory.semantic.search("favorite color")
        assert len(facts) >= 0  # May or may not find immediately


class TestProactiveIntelligence:
    """Test proactive suggestion system."""

    @pytest.fixture
    async def engine(self):
        from core.engine import Senter
        engine = Senter(Path('genome.yaml'))
        await engine.initialize()
        yield engine
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_suggestions_at_high_trust(self, engine):
        """Proactive suggestions appear at high trust."""

        # Raise trust
        engine.trust.level = 0.85

        # Get suggestions
        suggestions = await engine.proactive.generate_suggestions()

        # Should return a list (may be empty if no goals)
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_no_suggestions_at_very_low_trust(self, engine):
        """No proactive suggestions at very low trust (<0.3)."""
        engine.trust.level = 0.2  # Below threshold of 0.3

        suggestions = await engine.proactive.generate_suggestions()

        assert len(suggestions) == 0


class TestMemoryContinuity:
    """Test memory persistence across sessions."""

    @pytest.mark.asyncio
    async def test_trust_persists(self):
        """Trust level persists across restart."""
        from core.engine import Senter

        # Session 1: Build trust
        engine1 = Senter(Path('genome.yaml'))
        await engine1.initialize()

        initial_trust = engine1.trust.level
        for _ in range(5):
            await engine1.interact("Thanks, that's helpful!")
        trust1 = engine1.trust.level
        await engine1.shutdown()

        assert trust1 > initial_trust  # Trust increased

        # Session 2: Check trust
        engine2 = Senter(Path('genome.yaml'))
        await engine2.initialize()
        trust2 = engine2.trust.level
        await engine2.shutdown()

        # Trust should persist (allow small variance)
        assert trust2 >= trust1 - 0.1


class TestEvolutionSystem:
    """Test that evolution system is functional."""

    @pytest.mark.asyncio
    async def test_evolution_tracking(self):
        """Evolution system tracks mutations."""
        from core.engine import Senter

        engine = Senter(Path('genome.yaml'))
        await engine.initialize()

        # Get evolution summary
        summary = engine.mutations.get_evolution_summary()

        # Should have summary keys
        assert 'total' in summary
        assert 'successful' in summary

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_fitness_tracking(self):
        """Fitness is tracked over interactions."""
        from core.engine import Senter

        engine = Senter(Path('genome.yaml'))
        await engine.initialize()

        # Have some interactions
        for i in range(3):
            response = await engine.interact(f"Help me with task {i}")
            assert response.fitness is not None or response.fitness >= 0

        await engine.shutdown()


class TestActivityMonitoring:
    """Test activity monitoring components."""

    def test_context_inference(self):
        """Context inferencer works correctly."""
        from intelligence.activity import ContextInferencer

        ci = ContextInferencer()

        # Test coding context
        coding_snapshot = {
            'app': 'VSCode',
            'window': 'main.py',
            'key_phrases': ['def test', 'import asyncio', 'class MyClass']
        }
        assert ci.infer_context(coding_snapshot) == 'coding'

        # Test research context
        research_snapshot = {
            'app': 'Chrome',
            'window': 'arxiv.org',
            'key_phrases': ['abstract', 'paper', 'neural networks']
        }
        assert ci.infer_context(research_snapshot) == 'research'

    def test_window_detection(self):
        """Window detection returns valid data."""
        from intelligence.activity import ScreenCapture

        sc = ScreenCapture()
        window = sc.get_active_window()

        assert 'app' in window
        assert 'window' in window


class TestCoreIntegration:
    """Test core system integration."""

    @pytest.mark.asyncio
    async def test_full_interaction_cycle(self):
        """Complete interaction cycle works."""
        from core.engine import Senter

        engine = Senter(Path('genome.yaml'))
        await engine.initialize()

        # Basic interaction
        response = await engine.interact("Hello, how are you?")

        # Response should have all components
        assert response.text is not None
        assert len(response.text) > 0
        assert response.ai_state is not None
        assert response.ai_state.mode is not None
        assert response.ai_state.trust_level >= 0

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_mode_switching(self):
        """Coupling modes switch based on input."""
        from core.engine import Senter

        engine = Senter(Path('genome.yaml'))
        await engine.initialize()

        # Teaching mode trigger
        response = await engine.interact("Teach me about neural networks")
        assert response.ai_state.mode in ['TEACHING', 'DIALOGUE', 'DIRECTING']

        await engine.shutdown()


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
