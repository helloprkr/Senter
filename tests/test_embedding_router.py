#!/usr/bin/env python3
"""
Tests for EmbeddingRouter (CG-007)
Tests the semantic routing with top-N selection and confidence scoring.
"""

import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cosine_similarity():
    """Test cosine similarity calculation"""
    from Functions.embedding_router import cosine_similarity

    # Identical vectors should have similarity 1.0
    vec1 = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec1, vec1) - 1.0) < 0.001

    # Orthogonal vectors should have similarity 0.0
    vec2 = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(vec1, vec2)) < 0.001

    # Opposite vectors should have similarity -1.0
    vec3 = [-1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec1, vec3) - (-1.0)) < 0.001

    # Zero vectors should return 0.0
    vec_zero = [0.0, 0.0, 0.0]
    assert cosine_similarity(vec1, vec_zero) == 0.0

    return True


def test_embedding_router_init():
    """Test EmbeddingRouter initialization"""
    from Functions.embedding_router import EmbeddingRouter

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"
        focuses_dir.mkdir(parents=True)

        # Create a test focus
        test_focus = focuses_dir / "coding"
        test_focus.mkdir()
        senter_md = test_focus / "SENTER.md"
        senter_md.write_text("""---
system_prompt: |
  You are a helpful coding assistant.
---
# Coding Focus
Helps with programming tasks.
""")

        router = EmbeddingRouter(senter_root)

        # Should detect the focus
        assert "coding" in router.focus_descriptions

        return True


def test_get_focus_description():
    """Test focus description extraction from SENTER.md"""
    from Functions.embedding_router import get_focus_description

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"

        # Create test focus with system_prompt
        focus_dir = focuses_dir / "research"
        focus_dir.mkdir(parents=True)
        senter_md = focus_dir / "SENTER.md"
        senter_md.write_text("""---
system_prompt: |
  You help with research and information gathering.
  You can search the web and analyze data.
---
# Research Focus
""")

        desc = get_focus_description("research", senter_root)
        assert "research" in desc.lower()

        # Test missing focus
        desc = get_focus_description("nonexistent", senter_root)
        assert desc == "nonexistent"

        return True


def test_route_query_fallback():
    """Test that route_query falls back to 'general' when no embeddings"""
    from Functions.embedding_router import EmbeddingRouter

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"
        focuses_dir.mkdir(parents=True)

        router = EmbeddingRouter(senter_root)

        # With no embeddings, should fallback to 'general'
        focus, score, all_scores = router.route_query("test query")
        assert focus == "general"
        assert score == 0.0
        assert all_scores == {}

        return True


def test_top_n_focuses_fallback():
    """Test get_top_n_focuses fallback when no embeddings (CG-007)"""
    from Functions.embedding_router import EmbeddingRouter

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"
        focuses_dir.mkdir(parents=True)

        router = EmbeddingRouter(senter_root)

        # With no embeddings, should return general
        top_n = router.get_top_n_focuses("test query", n=3)
        assert len(top_n) == 1
        assert top_n[0][0] == "general"
        assert top_n[0][1] == 0.0

        return True


def test_route_with_confidence_structure():
    """Test route_with_confidence returns proper structure (CG-007)"""
    from Functions.embedding_router import EmbeddingRouter

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"
        focuses_dir.mkdir(parents=True)

        router = EmbeddingRouter(senter_root)

        result = router.route_with_confidence("test query")

        # Verify structure
        assert "selected_focus" in result
        assert "confidence" in result
        assert "top_3" in result
        assert "all_scores" in result
        assert "above_threshold" in result
        assert "fallback_used" in result

        # Verify types
        assert isinstance(result["selected_focus"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["top_3"], list)
        assert isinstance(result["above_threshold"], bool)
        assert isinstance(result["fallback_used"], bool)

        return True


def test_explain_routing():
    """Test explain_routing output format"""
    from Functions.embedding_router import EmbeddingRouter

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"
        focuses_dir.mkdir(parents=True)

        router = EmbeddingRouter(senter_root)

        explanation = router.explain_routing("test query")

        # Should include query and routing info
        assert "Query:" in explanation
        assert "Routed to:" in explanation

        return True


def test_cache_loading_and_saving():
    """Test embedding cache persistence"""
    from Functions.embedding_router import EmbeddingRouter, compute_content_hash, get_focus_description

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"
        focuses_dir.mkdir(parents=True)
        data_dir = senter_root / "data"
        data_dir.mkdir(parents=True)

        # Create test focus
        focus_dir = focuses_dir / "test_focus"
        focus_dir.mkdir()
        senter_md = focus_dir / "SENTER.md"
        senter_md.write_text("---\nsystem_prompt: Test\n---")

        # Get actual description as router would compute it
        desc = get_focus_description("test_focus", senter_root)

        # Create mock cache with matching content hash
        cache_file = senter_root / "data" / "focus_embeddings.json"
        cache_data = {
            "content_hash": compute_content_hash({"test_focus": desc}),
            "embeddings": {"test_focus": [0.1, 0.2, 0.3]},
            "descriptions": {"test_focus": desc}
        }
        cache_file.write_text(json.dumps(cache_data))

        # Router should load from cache
        router = EmbeddingRouter(senter_root)

        # Should have loaded cached embedding
        assert "test_focus" in router.focus_embeddings
        assert router.focus_embeddings["test_focus"] == [0.1, 0.2, 0.3]

        return True


def test_refresh_embeddings():
    """Test refresh_embeddings clears cache"""
    from Functions.embedding_router import EmbeddingRouter

    with tempfile.TemporaryDirectory() as tmpdir:
        senter_root = Path(tmpdir)
        focuses_dir = senter_root / "Focuses"
        focuses_dir.mkdir(parents=True)
        data_dir = senter_root / "data"
        data_dir.mkdir(parents=True)

        # Create cache file
        cache_file = senter_root / "data" / "focus_embeddings.json"
        cache_file.write_text('{"test": "data"}')

        router = EmbeddingRouter(senter_root)

        # Refresh should remove cache
        router.refresh_embeddings()

        # Cache file should be deleted
        assert not cache_file.exists()

        return True


if __name__ == "__main__":
    tests = [
        test_cosine_similarity,
        test_embedding_router_init,
        test_get_focus_description,
        test_route_query_fallback,
        test_top_n_focuses_fallback,
        test_route_with_confidence_structure,
        test_explain_routing,
        test_cache_loading_and_saving,
        test_refresh_embeddings,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")
