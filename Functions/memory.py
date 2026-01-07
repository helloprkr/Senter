#!/usr/bin/env python3
"""
Conversation Memory System for Senter
Persistent memory with vector search for relevant context retrieval
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger("senter.memory")

# Try to import embedding function
try:
    from embedding_router import embed_text
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from Functions.embedding_router import embed_text
        EMBEDDINGS_AVAILABLE = True
    except ImportError:
        EMBEDDINGS_AVAILABLE = False
        logger.warning("Embedding module not available for memory search")


@dataclass
class ConversationChunk:
    """A searchable chunk of conversation"""
    conversation_id: str
    chunk_index: int
    role: str
    content: str
    timestamp: str
    focus: str = "general"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationChunk":
        return cls(**data)


@dataclass
class Conversation:
    """A complete conversation session"""
    id: str
    messages: list[dict]
    focus: str
    created: str
    summary: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(**data)


class ConversationMemory:
    """Persistent conversation memory with vector search"""

    def __init__(self, senter_root: Path = None):
        self.senter_root = senter_root or Path(".")
        self.conversations_dir = self.senter_root / "data" / "conversations"
        self.index_path = self.conversations_dir / "index.json"
        self.embeddings_path = self.conversations_dir / "embeddings.npy"
        self.chunks_path = self.conversations_dir / "chunks.json"

        # Ensure directories exist
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize index
        self.index = self._load_index()
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()

        logger.info(f"Memory initialized: {len(self.index['conversations'])} conversations, {len(self.chunks)} chunks")

    def _load_index(self) -> dict:
        """Load conversation index"""
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text())
            except Exception as e:
                logger.warning(f"Error loading index: {e}")

        return {
            "conversations": [],
            "last_id": 0
        }

    def _save_index(self):
        """Save conversation index"""
        try:
            self.index_path.write_text(json.dumps(self.index, indent=2))
        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def _load_chunks(self) -> list[dict]:
        """Load all conversation chunks"""
        if self.chunks_path.exists():
            try:
                return json.loads(self.chunks_path.read_text())
            except Exception as e:
                logger.warning(f"Error loading chunks: {e}")
        return []

    def _save_chunks(self):
        """Save all chunks"""
        try:
            self.chunks_path.write_text(json.dumps(self.chunks, indent=2))
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")

    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings numpy array"""
        if self.embeddings_path.exists():
            try:
                return np.load(self.embeddings_path)
            except Exception as e:
                logger.warning(f"Error loading embeddings: {e}")
        return None

    def _save_embeddings(self, embeddings: np.ndarray):
        """Save embeddings numpy array"""
        try:
            np.save(self.embeddings_path, embeddings)
            self.embeddings = embeddings
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def _generate_id(self) -> str:
        """Generate unique conversation ID"""
        self.index["last_id"] += 1
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{date_str}_{self.index['last_id']:04d}"

    def save_conversation(self, messages: list[dict], focus: str = "general") -> str:
        """
        Save a conversation and generate embeddings for search.

        Args:
            messages: List of message dicts with 'role' and 'content'
            focus: The focus used during conversation

        Returns:
            Conversation ID
        """
        if not messages:
            return ""

        # Create conversation
        conv_id = self._generate_id()
        timestamp = datetime.now().isoformat()

        conversation = Conversation(
            id=conv_id,
            messages=messages,
            focus=focus,
            created=timestamp,
            summary=self._create_summary(messages)
        )

        # Save conversation file
        conv_file = self.conversations_dir / f"{conv_id}.json"
        try:
            conv_file.write_text(json.dumps(conversation.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Error saving conversation file: {e}")
            return ""

        # Add to index
        self.index["conversations"].append({
            "id": conv_id,
            "focus": focus,
            "created": timestamp,
            "message_count": len(messages),
            "summary": conversation.summary
        })
        self._save_index()

        # Create and embed chunks
        new_chunks = []
        new_embeddings = []

        for i, msg in enumerate(messages):
            if not msg.get("content"):
                continue

            chunk = ConversationChunk(
                conversation_id=conv_id,
                chunk_index=i,
                role=msg.get("role", "unknown"),
                content=msg["content"],
                timestamp=timestamp,
                focus=focus
            )
            new_chunks.append(chunk.to_dict())

            # Generate embedding if available
            if EMBEDDINGS_AVAILABLE:
                embedding = embed_text(msg["content"][:500])  # Limit content length
                if embedding:
                    new_embeddings.append(embedding)
                else:
                    # Use zero vector as placeholder
                    new_embeddings.append([0.0] * 768)

        # Add chunks to store
        self.chunks.extend(new_chunks)
        self._save_chunks()

        # Update embeddings
        if new_embeddings:
            new_emb_array = np.array(new_embeddings)
            if self.embeddings is not None:
                combined = np.vstack([self.embeddings, new_emb_array])
            else:
                combined = new_emb_array
            self._save_embeddings(combined)

        logger.info(f"Saved conversation {conv_id} with {len(messages)} messages")
        return conv_id

    def _create_summary(self, messages: list[dict], max_length: int = 100) -> str:
        """Create a brief summary of the conversation"""
        if not messages:
            return ""

        # Get first user message as summary basis
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                content = msg["content"][:max_length]
                if len(msg["content"]) > max_length:
                    content += "..."
                return content

        return "Conversation"

    def search_memory(self, query: str, limit: int = 5, threshold: float = 0.3) -> list[ConversationChunk]:
        """
        Search conversation memory for relevant context.

        Args:
            query: Search query
            limit: Maximum results to return
            threshold: Minimum similarity threshold

        Returns:
            List of relevant conversation chunks
        """
        if not self.chunks:
            return []

        if not EMBEDDINGS_AVAILABLE or self.embeddings is None:
            # Fall back to keyword search
            return self._keyword_search(query, limit)

        # Embed query
        query_embedding = embed_text(query)
        if not query_embedding:
            return self._keyword_search(query, limit)

        query_vec = np.array(query_embedding)

        # Calculate cosine similarities
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)

        # Avoid division by zero
        valid_mask = (norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(self.embeddings))
        similarities[valid_mask] = np.dot(
            self.embeddings[valid_mask], query_vec
        ) / (norms[valid_mask] * query_norm)

        # Get top results above threshold
        indices = np.argsort(similarities)[::-1]
        results = []

        for idx in indices[:limit * 2]:  # Get more to filter
            if similarities[idx] < threshold:
                break
            if idx < len(self.chunks):
                chunk = ConversationChunk.from_dict(self.chunks[idx])
                chunk.similarity = similarities[idx]  # Add similarity score
                results.append(chunk)
                if len(results) >= limit:
                    break

        logger.info(f"Memory search for '{query[:30]}...' found {len(results)} results")
        return results

    def _keyword_search(self, query: str, limit: int) -> list[ConversationChunk]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_chunks = []
        for chunk_data in self.chunks:
            content_lower = chunk_data["content"].lower()
            # Simple word overlap scoring
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored_chunks.append((overlap, chunk_data))

        # Sort by score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, chunk_data in scored_chunks[:limit]:
            results.append(ConversationChunk.from_dict(chunk_data))

        return results

    def get_recent_conversations(self, n: int = 10) -> list[Conversation]:
        """Get N most recent conversations"""
        recent_ids = [
            c["id"] for c in sorted(
                self.index["conversations"],
                key=lambda x: x["created"],
                reverse=True
            )[:n]
        ]

        conversations = []
        for conv_id in recent_ids:
            conv = self.load_conversation(conv_id)
            if conv:
                conversations.append(conv)

        return conversations

    def load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a specific conversation by ID"""
        conv_file = self.conversations_dir / f"{conv_id}.json"
        if not conv_file.exists():
            return None

        try:
            data = json.loads(conv_file.read_text())
            return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading conversation {conv_id}: {e}")
            return None

    def get_context_for_query(self, query: str, max_context: int = 3) -> str:
        """
        Get relevant context from memory for a query.
        Returns formatted string for inclusion in system prompt.
        """
        results = self.search_memory(query, limit=max_context)

        if not results:
            return ""

        context_parts = ["[Relevant past conversations:]"]

        for chunk in results:
            date_str = chunk.timestamp[:10] if chunk.timestamp else "unknown"
            context_parts.append(
                f"- [{date_str}] {chunk.role}: {chunk.content[:200]}..."
                if len(chunk.content) > 200 else
                f"- [{date_str}] {chunk.role}: {chunk.content}"
            )

        return "\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get memory statistics"""
        return {
            "total_conversations": len(self.index["conversations"]),
            "total_chunks": len(self.chunks),
            "has_embeddings": self.embeddings is not None,
            "embedding_count": len(self.embeddings) if self.embeddings is not None else 0
        }


# Convenience functions
def save_conversation(messages: list[dict], focus: str = "general", senter_root: Path = None) -> str:
    """Save a conversation to memory"""
    memory = ConversationMemory(senter_root)
    return memory.save_conversation(messages, focus)


def search_memory(query: str, limit: int = 5, senter_root: Path = None) -> list[ConversationChunk]:
    """Search conversation memory"""
    memory = ConversationMemory(senter_root)
    return memory.search_memory(query, limit)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senter Conversation Memory")
    parser.add_argument("--search", "-s", help="Search memory for query")
    parser.add_argument("--recent", "-r", type=int, help="Show N recent conversations")
    parser.add_argument("--stats", action="store_true", help="Show memory statistics")
    parser.add_argument("--test", action="store_true", help="Run test save/search")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    memory = ConversationMemory(Path("."))

    if args.stats:
        stats = memory.get_stats()
        print("\nMemory Statistics:")
        print(f"  Conversations: {stats['total_conversations']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Embeddings: {stats['embedding_count']}")

    elif args.search:
        print(f"\nSearching for: '{args.search}'")
        results = memory.search_memory(args.search)
        if results:
            print(f"\nFound {len(results)} results:")
            for i, chunk in enumerate(results, 1):
                score = getattr(chunk, 'similarity', 'N/A')
                print(f"\n{i}. [{chunk.conversation_id}] {chunk.role}")
                print(f"   Score: {score:.3f}" if isinstance(score, float) else f"   Score: {score}")
                print(f"   Content: {chunk.content[:100]}...")
        else:
            print("No results found")

    elif args.recent:
        conversations = memory.get_recent_conversations(args.recent)
        print(f"\nRecent Conversations ({len(conversations)}):")
        for conv in conversations:
            print(f"\n  [{conv.id}] Focus: {conv.focus}")
            print(f"  Created: {conv.created}")
            print(f"  Summary: {conv.summary}")

    elif args.test:
        print("\nRunning memory test...")

        # Save test conversation
        test_messages = [
            {"role": "user", "content": "What's the best way to learn Python programming?"},
            {"role": "assistant", "content": "The best way to learn Python is to start with the basics: variables, loops, and functions. Then build small projects to practice."},
            {"role": "user", "content": "Can you recommend some beginner projects?"},
            {"role": "assistant", "content": "Sure! Try building a calculator, a to-do list app, or a simple game like number guessing. These help solidify fundamentals."}
        ]

        conv_id = memory.save_conversation(test_messages, focus="coding")
        print(f"  Saved test conversation: {conv_id}")

        # Search for it
        results = memory.search_memory("learning Python")
        print(f"  Search for 'learning Python' found {len(results)} results")

        if results:
            print(f"  Top result: {results[0].content[:50]}...")

        print("\nTest complete!")

    else:
        parser.print_help()
