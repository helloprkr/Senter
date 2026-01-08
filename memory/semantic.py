"""
Semantic Memory - Facts and concepts storage.

Stores what the AI knows about the world and the user.
Supports embedding-based semantic search.
"""

from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.embeddings import EmbeddingModel


class SemanticMemory:
    """
    Semantic memory for facts and concepts.

    Stores:
    - User-stated facts ("remember that...")
    - Learned facts from conversations
    - World knowledge

    Supports:
    - Text search
    - Embedding-based semantic search
    - Decay over time for relevance
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: Dict,
        embeddings: Optional["EmbeddingModel"] = None,
    ):
        self.conn = conn
        self.config = config
        self.decay_rate = config.get("decay_rate", 0.001)
        self.embeddings = embeddings
        self._use_embeddings = embeddings is not None and config.get("embedding_search", True)

    def store(
        self,
        content: str,
        domain: str = "general",
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Store a semantic fact with embedding.

        Args:
            content: The fact content
            domain: Category (user_context, project_context, world_knowledge)
            embedding: Optional pre-computed embedding vector

        Returns:
            ID of stored fact
        """
        fact_id = str(uuid.uuid4())[:8]

        embedding_blob = None

        # Generate embedding if we have an embedding model
        if embedding is not None:
            # Use provided embedding
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        elif self._use_embeddings and self.embeddings is not None:
            # Generate embedding from content
            try:
                import asyncio

                # Get or create event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, create a task
                    future = asyncio.ensure_future(self.embeddings.embed(content))
                    # This is synchronous context within async - use sync store_async instead
                    embedding_blob = None  # Will generate in async path
                except RuntimeError:
                    # No running loop - create one
                    loop = asyncio.new_event_loop()
                    try:
                        embedding_vec = loop.run_until_complete(self.embeddings.embed(content))
                        embedding_blob = np.array(embedding_vec, dtype=np.float32).tobytes()
                    finally:
                        loop.close()
            except Exception as e:
                # Fallback: store without embedding
                print(f"Warning: Could not generate embedding: {e}")
                embedding_blob = None

        self.conn.execute(
            """
            INSERT INTO semantic (id, content, domain, embedding, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (fact_id, content, domain, embedding_blob, datetime.now().isoformat(), 0),
        )
        self.conn.commit()

        return fact_id

    async def store_async(
        self,
        content: str,
        domain: str = "general",
    ) -> str:
        """
        Async version of store that properly awaits embedding generation.

        Args:
            content: The fact content
            domain: Category

        Returns:
            ID of stored fact
        """
        fact_id = str(uuid.uuid4())[:8]

        embedding_blob = None

        # Generate embedding if we have an embedding model
        if self._use_embeddings and self.embeddings is not None:
            try:
                embedding_vec = await self.embeddings.embed(content)
                embedding_blob = np.array(embedding_vec, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"Warning: Could not generate embedding: {e}")

        self.conn.execute(
            """
            INSERT INTO semantic (id, content, domain, embedding, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (fact_id, content, domain, embedding_blob, datetime.now().isoformat(), 0),
        )
        self.conn.commit()

        return fact_id

    def search(
        self,
        query: str,
        limit: int = 5,
        domain: Optional[str] = None,
        threshold: float = 0.15,  # Lowered from 0.3 - be more inclusive
    ) -> List[Dict]:
        """
        Search for relevant facts using semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            domain: Filter by domain
            threshold: Minimum similarity threshold for embedding search

        Returns:
            List of matching facts with similarity scores
        """
        # Try embedding-based search first if available
        if self._use_embeddings and self.embeddings is not None:
            results = self._search_by_embedding(query, limit, domain, threshold)
            if results:
                return results
            # Fall through to keyword search if no embedding results

        # Fall back to keyword-based search
        return self._search_by_keywords(query, limit, domain)

    async def search_async(
        self,
        query: str,
        limit: int = 5,
        domain: Optional[str] = None,
        threshold: float = 0.15,  # Lowered from 0.3 - be more inclusive
    ) -> List[Dict]:
        """
        Async version of search that properly awaits embedding generation.
        """
        if self._use_embeddings and self.embeddings is not None:
            results = await self._search_by_embedding_async(query, limit, domain, threshold)
            if results:
                return results

        return self._search_by_keywords(query, limit, domain)

    def _search_by_embedding(
        self,
        query: str,
        limit: int,
        domain: Optional[str],
        threshold: float,
    ) -> List[Dict]:
        """Search using embedding similarity (sync version)."""
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                # Can't run sync in async context - return empty to fall back
                return []
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        self._search_by_embedding_async(query, limit, domain, threshold)
                    )
                finally:
                    loop.close()
        except Exception as e:
            print(f"Warning: Embedding search failed: {e}")
            return []

    async def _search_by_embedding_async(
        self,
        query: str,
        limit: int,
        domain: Optional[str],
        threshold: float,
    ) -> List[Dict]:
        """Search using embedding similarity (async version)."""
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed(query)
            query_vec = np.array(query_embedding, dtype=np.float32)

            # Get all candidates from database
            if domain:
                cursor = self.conn.execute(
                    "SELECT id, content, domain, embedding, decay_factor FROM semantic WHERE domain = ? AND embedding IS NOT NULL",
                    (domain,),
                )
            else:
                cursor = self.conn.execute(
                    "SELECT id, content, domain, embedding, decay_factor FROM semantic WHERE embedding IS NOT NULL"
                )

            results = []
            for row in cursor.fetchall():
                if row["embedding"]:
                    # Decode stored embedding
                    stored_vec = np.frombuffer(row["embedding"], dtype=np.float32)

                    # Compute cosine similarity
                    similarity = self._cosine_similarity_np(query_vec, stored_vec)

                    if similarity >= threshold:
                        results.append(
                            {
                                "id": row["id"],
                                "content": row["content"],
                                "domain": row["domain"],
                                "similarity": float(similarity),
                                "relevance": float(row["decay_factor"] * similarity),
                            }
                        )

            # Sort by relevance (decay_factor * similarity) and limit
            results.sort(key=lambda x: x["relevance"], reverse=True)
            results = results[:limit]

            # Update access counts
            for r in results:
                self._update_access(r["id"])

            return results

        except Exception as e:
            print(f"Warning: Embedding search failed: {e}")
            return []

    def _cosine_similarity_np(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity using numpy."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _search_by_keywords(
        self,
        query: str,
        limit: int,
        domain: Optional[str],
    ) -> List[Dict]:
        """Fallback keyword-based search."""
        keywords = self._extract_keywords(query)

        if not keywords:
            # Fall back to domain-based retrieval
            if domain:
                return self.get_by_domain(domain, limit)
            cursor = self.conn.execute(
                "SELECT id, content, domain, access_count, decay_factor FROM semantic ORDER BY access_count DESC LIMIT ?",
                (limit,),
            )
        else:
            # Build OR query for keyword matching
            conditions = " OR ".join(["content LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords]

            if domain:
                conditions = f"({conditions}) AND domain = ?"
                params.append(domain)

            params.append(limit)

            cursor = self.conn.execute(
                f"""
                SELECT id, content, domain, access_count, decay_factor
                FROM semantic
                WHERE {conditions}
                ORDER BY access_count DESC, decay_factor DESC
                LIMIT ?
            """,
                params,
            )

        results = []
        for row in cursor.fetchall():
            # Update access count and timestamp
            self._update_access(row["id"])

            results.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "domain": row["domain"],
                    "relevance": row["decay_factor"],
                }
            )

        return results

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from a query."""
        # Common words to ignore - BUT keep important identity words!
        stopwords = {
            "what", "is", "the", "a", "an", "and", "or", "do", "did", "does",
            "you", "me", "we", "they", "it", "this", "that", "of", "to", "in",
            "on", "at", "for", "with", "about", "how", "when", "where", "who", "which",
            "can", "could", "would", "should", "will", "be", "have", "has", "had",
            "remember", "recall", "tell", "think", "please", "just", "also"
        }
        # NOTE: Removed "my", "i", "know", "name" - these are important for identity queries!

        # Extract words, filter stopwords and short words
        words = query.lower().split()
        keywords = [w.strip("?!.,") for w in words if w.lower() not in stopwords and len(w) > 2]

        return keywords[:5]  # Limit to top 5 keywords

    def search_by_embedding(
        self,
        embedding: List[float],
        limit: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Search using embedding similarity.

        Args:
            embedding: Query embedding vector
            limit: Maximum results
            threshold: Minimum similarity threshold

        Returns:
            List of matching facts with similarity scores
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, domain, embedding, decay_factor
            FROM semantic
            WHERE embedding IS NOT NULL
        """
        )

        results = []
        for row in cursor.fetchall():
            if row["embedding"]:
                stored_embedding = json.loads(row["embedding"].decode())
                similarity = self._cosine_similarity(embedding, stored_embedding)

                if similarity >= threshold:
                    results.append(
                        {
                            "id": row["id"],
                            "content": row["content"],
                            "domain": row["domain"],
                            "similarity": similarity,
                            "relevance": row["decay_factor"] * similarity,
                        }
                    )

        # Sort by relevance and limit
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]

    def get_by_domain(self, domain: str, limit: int = 10) -> List[Dict]:
        """Get facts by domain."""
        cursor = self.conn.execute(
            """
            SELECT id, content, domain, access_count
            FROM semantic
            WHERE domain = ?
            ORDER BY access_count DESC
            LIMIT ?
        """,
            (domain, limit),
        )

        return [dict(row) for row in cursor.fetchall()]

    def delete(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        cursor = self.conn.execute("DELETE FROM semantic WHERE id = ?", (fact_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def apply_decay(self) -> int:
        """
        Apply decay to all facts.

        Returns:
            Number of facts updated
        """
        cursor = self.conn.execute(
            """
            UPDATE semantic
            SET decay_factor = decay_factor * (1 - ?)
            WHERE decay_factor > 0.01
        """,
            (self.decay_rate,),
        )
        self.conn.commit()
        return cursor.rowcount

    def _update_access(self, fact_id: str) -> None:
        """Update access count and timestamp."""
        self.conn.execute(
            """
            UPDATE semantic
            SET access_count = access_count + 1,
                last_accessed = ?,
                decay_factor = MIN(decay_factor * 1.1, 1.0)
            WHERE id = ?
        """,
            (datetime.now().isoformat(), fact_id),
        )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def count(self, domain: Optional[str] = None) -> int:
        """Count facts, optionally filtered by domain."""
        if domain:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM semantic WHERE domain = ?", (domain,)
            )
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM semantic")

        return cursor.fetchone()[0]
