#!/usr/bin/env python3
"""
Embedding-based Semantic Router for Senter
Routes queries to the most relevant Focus based on semantic similarity
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("senter.router")

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CACHE_FILE = "data/focus_embeddings.json"


def embed_text(text: str, model: str = EMBEDDING_MODEL) -> Optional[list[float]]:
    """
    Generate embedding for text using Ollama.

    Args:
        text: Text to embed
        model: Ollama embedding model name

    Returns:
        List of floats (embedding vector) or None on error
    """
    if not requests:
        logger.error("requests library not available")
        return None

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": text},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            # Ollama returns embeddings in 'embeddings' key (list of embeddings)
            embeddings = data.get("embeddings", [])
            if embeddings:
                return embeddings[0]  # Return first embedding
        else:
            logger.error(f"Embedding API error: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)

    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def get_focus_description(focus_name: str, senter_root: Path) -> str:
    """
    Extract description from Focus SENTER.md for embedding.

    Args:
        focus_name: Name of the focus
        senter_root: Root path of Senter

    Returns:
        Text description to embed
    """
    senter_md_path = senter_root / "Focuses" / focus_name / "SENTER.md"

    if not senter_md_path.exists():
        return focus_name

    try:
        content = senter_md_path.read_text()

        # Extract system_prompt from YAML frontmatter
        if "system_prompt:" in content:
            # Find system_prompt section
            lines = content.split("\n")
            prompt_lines = []
            in_prompt = False

            for line in lines:
                if "system_prompt:" in line:
                    in_prompt = True
                    # Get content after system_prompt: |
                    continue
                elif in_prompt:
                    if line.startswith("  ") or line.strip() == "":
                        prompt_lines.append(line.strip())
                    elif not line.startswith(" ") and line.strip() and ":" in line:
                        # New YAML key, end of prompt
                        break

            if prompt_lines:
                # Take first 500 chars of system prompt
                prompt_text = " ".join(prompt_lines)[:500]
                return f"{focus_name}: {prompt_text}"

        # Fallback: extract first paragraph after frontmatter
        if "---" in content:
            parts = content.split("---")
            if len(parts) >= 3:
                markdown = parts[2].strip()
                first_para = markdown.split("\n\n")[0][:300]
                return f"{focus_name}: {first_para}"

    except Exception as e:
        logger.warning(f"Error reading focus {focus_name}: {e}")

    return focus_name


def compute_content_hash(focus_descriptions: dict) -> str:
    """Compute hash of focus descriptions for cache invalidation."""
    content = json.dumps(focus_descriptions, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


class EmbeddingRouter:
    """Semantic router that uses embeddings to route queries to focuses."""

    def __init__(self, senter_root: Path = None):
        self.senter_root = senter_root or Path(".")
        self.focuses_dir = self.senter_root / "Focuses"
        self.cache_path = self.senter_root / CACHE_FILE

        # Focus embeddings: {focus_name: embedding_vector}
        self.focus_embeddings: dict[str, list[float]] = {}
        self.focus_descriptions: dict[str, str] = {}

        # Load or generate embeddings
        self._load_or_generate_embeddings()

    def _get_user_focuses(self) -> list[str]:
        """Get list of user-facing focuses (exclude internal)."""
        focuses = []
        if not self.focuses_dir.exists():
            return focuses

        for item in self.focuses_dir.iterdir():
            if item.is_dir() and item.name != "internal":
                senter_md = item / "SENTER.md"
                if senter_md.exists():
                    focuses.append(item.name)

        return focuses

    def _load_cache(self) -> Optional[dict]:
        """Load cached embeddings if valid."""
        if not self.cache_path.exists():
            return None

        try:
            cache = json.loads(self.cache_path.read_text())
            return cache
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
            return None

    def _save_cache(self, content_hash: str):
        """Save embeddings to cache."""
        cache_dir = self.cache_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache = {
            "content_hash": content_hash,
            "embeddings": self.focus_embeddings,
            "descriptions": self.focus_descriptions
        }

        try:
            self.cache_path.write_text(json.dumps(cache, indent=2))
            logger.info(f"Cached embeddings for {len(self.focus_embeddings)} focuses")
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def _load_or_generate_embeddings(self):
        """Load embeddings from cache or generate new ones."""
        focuses = self._get_user_focuses()
        if not focuses:
            logger.warning("No focuses found")
            return

        # Get descriptions for all focuses
        for focus in focuses:
            self.focus_descriptions[focus] = get_focus_description(focus, self.senter_root)

        # Check cache
        content_hash = compute_content_hash(self.focus_descriptions)
        cache = self._load_cache()

        if cache and cache.get("content_hash") == content_hash:
            # Cache is valid
            self.focus_embeddings = cache.get("embeddings", {})
            logger.info(f"Loaded cached embeddings for {len(self.focus_embeddings)} focuses")
            return

        # Generate new embeddings
        logger.info(f"Generating embeddings for {len(focuses)} focuses...")

        for focus, description in self.focus_descriptions.items():
            embedding = embed_text(description)
            if embedding:
                self.focus_embeddings[focus] = embedding
                logger.debug(f"  Embedded: {focus}")
            else:
                logger.warning(f"  Failed to embed: {focus}")

        # Save to cache
        self._save_cache(content_hash)

    def route_query(self, query: str, threshold: float = 0.3) -> tuple[str, float, dict]:
        """
        Route a query to the most relevant focus.

        Args:
            query: User query to route
            threshold: Minimum similarity threshold (default 0.3)

        Returns:
            Tuple of (focus_name, similarity_score, all_scores)
        """
        if not self.focus_embeddings:
            logger.warning("No focus embeddings available, defaulting to 'general'")
            return "general", 0.0, {}

        # Embed the query
        query_embedding = embed_text(query)
        if not query_embedding:
            logger.warning("Failed to embed query, defaulting to 'general'")
            return "general", 0.0, {}

        # Calculate similarities
        similarities = {}
        for focus, focus_embedding in self.focus_embeddings.items():
            sim = cosine_similarity(query_embedding, focus_embedding)
            similarities[focus] = sim

        # Find best match
        best_focus = max(similarities, key=similarities.get)
        best_score = similarities[best_focus]

        # Check threshold
        if best_score < threshold:
            logger.info(f"Best match '{best_focus}' ({best_score:.3f}) below threshold, using 'general'")
            return "general", best_score, similarities

        logger.info(f"Routed to '{best_focus}' with score {best_score:.3f}")
        return best_focus, best_score, similarities

    def get_top_n_focuses(self, query: str, n: int = 3) -> list[tuple[str, float]]:
        """
        Get top N matching focuses for a query (CG-007).

        Args:
            query: User query
            n: Number of top matches to return (default 3)

        Returns:
            List of (focus_name, similarity_score) tuples, sorted by score descending
        """
        if not self.focus_embeddings:
            logger.warning("No focus embeddings available")
            return [("general", 0.0)]

        # Embed the query
        query_embedding = embed_text(query)
        if not query_embedding:
            logger.warning("Failed to embed query")
            return [("general", 0.0)]

        # Calculate similarities
        similarities = []
        for focus, focus_embedding in self.focus_embeddings.items():
            sim = cosine_similarity(query_embedding, focus_embedding)
            similarities.append((focus, sim))

        # Sort by score descending and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_n = similarities[:n]

        logger.info(f"Top {n} focuses: {[(f, f'{s:.3f}') for f, s in top_n]}")
        return top_n

    def route_with_confidence(self, query: str, threshold: float = 0.3) -> dict:
        """
        Route query with detailed confidence information (CG-007).

        Returns dict with routing decision and metadata.
        """
        focus, score, all_scores = self.route_query(query, threshold)
        top_3 = self.get_top_n_focuses(query, n=3)

        return {
            "selected_focus": focus,
            "confidence": round(score, 4),
            "top_3": [{"focus": f, "score": round(s, 4)} for f, s in top_3],
            "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
            "above_threshold": score >= threshold,
            "fallback_used": focus == "general" and (not all_scores or max(all_scores.values()) < threshold)
        }

    def explain_routing(self, query: str) -> str:
        """Get human-readable explanation of routing decision."""
        focus, score, all_scores = self.route_query(query)

        # Sort by score
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        lines = [
            f"Query: \"{query}\"",
            f"Routed to: {focus} (score: {score:.3f})",
            "",
            "All scores:"
        ]

        for f, s in sorted_scores:
            marker = ">>>" if f == focus else "   "
            lines.append(f"  {marker} {f}: {s:.3f}")

        return "\n".join(lines)

    def refresh_embeddings(self):
        """Force regeneration of all embeddings."""
        if self.cache_path.exists():
            self.cache_path.unlink()
        self.focus_embeddings = {}
        self.focus_descriptions = {}
        self._load_or_generate_embeddings()


# Convenience function
def route_query(query: str, senter_root: Path = None) -> str:
    """Route a query to the best focus."""
    router = EmbeddingRouter(senter_root)
    focus, _, _ = router.route_query(query)
    return focus


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Router for Senter")
    parser.add_argument("query", nargs="?", help="Query to route")
    parser.add_argument("--explain", "-e", action="store_true", help="Show detailed routing explanation")
    parser.add_argument("--refresh", "-r", action="store_true", help="Refresh cached embeddings")
    parser.add_argument("--list", "-l", action="store_true", help="List loaded focuses")

    args = parser.parse_args()

    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    router = EmbeddingRouter(Path("."))

    if args.refresh:
        print("Refreshing embeddings...")
        router.refresh_embeddings()
        print(f"Done. {len(router.focus_embeddings)} focuses embedded.")

    elif args.list:
        print(f"\nLoaded focuses ({len(router.focus_embeddings)}):")
        for focus, desc in router.focus_descriptions.items():
            has_embedding = "Y" if focus in router.focus_embeddings else "N"
            print(f"  [{has_embedding}] {focus}")
            print(f"      {desc[:80]}...")
        print()

    elif args.query:
        if args.explain:
            print(router.explain_routing(args.query))
        else:
            focus, score, _ = router.route_query(args.query)
            print(f"Route: {focus} (score: {score:.3f})")

    else:
        # Interactive mode
        print("\nSemantic Router - Interactive Mode")
        print("Enter queries to test routing (Ctrl+C to exit)\n")

        while True:
            try:
                query = input("Query: ").strip()
                if query:
                    print(router.explain_routing(query))
                    print()
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except EOFError:
                break
