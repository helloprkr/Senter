"""
Knowledge Graph - Hierarchical structured knowledge.

Stores entities, relationships, and context in a graph structure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.living_memory import LivingMemory
    from models.embeddings import EmbeddingModel


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str
    node_type: str  # entity | concept | fact | relation
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.node_type,
            "name": self.name,
            "properties": self.properties,
        }


@dataclass
class KnowledgeEdge:
    """An edge (relationship) in the knowledge graph."""

    source_id: str
    target_id: str
    relation: str  # is_a | has_property | related_to | etc.
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Hierarchical knowledge graph.

    Stores structured knowledge with relationships between entities.
    """

    def __init__(
        self,
        config: Dict = None,
        memory: Optional["LivingMemory"] = None,
        embeddings: Optional["EmbeddingModel"] = None,
    ):
        config = config or {}
        self.config = config
        self.memory = memory
        self.embeddings = embeddings

        # Graph storage
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: List[KnowledgeEdge] = []

        # Domain organization
        self.domains = config.get("domains", [])

        # Index for fast lookup
        self._type_index: Dict[str, List[str]] = {}
        self._name_index: Dict[str, str] = {}

    def add_node(
        self,
        node_type: str,
        name: str,
        properties: Dict[str, Any] = None,
        node_id: str = None,
    ) -> KnowledgeNode:
        """
        Add a node to the graph.

        Args:
            node_type: Type of node (entity, concept, fact)
            name: Node name
            properties: Additional properties
            node_id: Optional ID (auto-generated if not provided)

        Returns:
            Created node
        """
        import uuid

        node_id = node_id or str(uuid.uuid4())[:8]

        node = KnowledgeNode(
            id=node_id,
            node_type=node_type,
            name=name,
            properties=properties or {},
        )

        self._nodes[node_id] = node

        # Update indexes
        if node_type not in self._type_index:
            self._type_index[node_type] = []
        self._type_index[node_type].append(node_id)
        self._name_index[name.lower()] = node_id

        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        properties: Dict[str, Any] = None,
    ) -> Optional[KnowledgeEdge]:
        """
        Add an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relationship type
            weight: Edge weight
            properties: Additional properties

        Returns:
            Created edge or None if nodes don't exist
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            properties=properties or {},
        )

        self._edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        node = self._nodes.get(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = datetime.now()
        return node

    def find_by_name(self, name: str) -> Optional[KnowledgeNode]:
        """Find a node by name."""
        if not isinstance(name, str):
            return None
        node_id = self._name_index.get(name.lower())
        if node_id:
            return self.get_node(node_id)
        return None

    def find_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Find all nodes of a type."""
        node_ids = self._type_index.get(node_type, [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_related(
        self,
        node_id: str,
        relation: str = None,
        direction: str = "outgoing",
    ) -> List[KnowledgeNode]:
        """
        Get nodes related to a given node.

        Args:
            node_id: Source node ID
            relation: Filter by relation type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of related nodes
        """
        related = []

        for edge in self._edges:
            if direction in ("outgoing", "both") and edge.source_id == node_id:
                if relation is None or edge.relation == relation:
                    if edge.target_id in self._nodes:
                        related.append(self._nodes[edge.target_id])

            if direction in ("incoming", "both") and edge.target_id == node_id:
                if relation is None or edge.relation == relation:
                    if edge.source_id in self._nodes:
                        related.append(self._nodes[edge.source_id])

        return related

    def query(
        self,
        intent: Dict[str, Any],
        scope: List[str] = None,
        mode: Any = None,
    ) -> "QueryResult":
        """
        Query the knowledge graph based on intent.

        Args:
            intent: Parsed intent
            scope: Capability scope
            mode: Coupling mode

        Returns:
            Query result with knowledge and memory context
        """
        results = []

        # Extract entities from intent
        entities = intent.get("entities", [])

        # Search for each entity
        for entity in entities:
            # Handle both string and dict entities
            entity_name = entity if isinstance(entity, str) else entity.get("name", str(entity))
            node = self.find_by_name(entity_name)
            if node:
                results.append(node.to_dict())

                # Get related nodes
                related = self.get_related(node.id)
                for rel in related[:3]:  # Limit related nodes
                    results.append(rel.to_dict())

        # Text search in properties
        intent_text = intent.get("intent", intent.get("raw_input", ""))
        for node in self._nodes.values():
            if intent_text.lower() in node.name.lower():
                if node.to_dict() not in results:
                    results.append(node.to_dict())

        return QueryResult(
            knowledge=results[:10],  # Limit results
            memory=None,  # To be filled by caller
        )

    def search(self, query: str, limit: int = 10) -> List[KnowledgeNode]:
        """Simple text search across nodes."""
        query_lower = query.lower()
        results = []

        for node in self._nodes.values():
            if query_lower in node.name.lower():
                results.append(node)

            # Search in properties
            for value in node.properties.values():
                if isinstance(value, str) and query_lower in value.lower():
                    if node not in results:
                        results.append(node)

        # Sort by access count
        results.sort(key=lambda n: n.access_count, reverse=True)
        return results[:limit]

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]

        # Remove from indexes
        if node.node_type in self._type_index:
            self._type_index[node.node_type] = [
                nid for nid in self._type_index[node.node_type] if nid != node_id
            ]

        if node.name.lower() in self._name_index:
            del self._name_index[node.name.lower()]

        # Remove edges
        self._edges = [
            e for e in self._edges if e.source_id != node_id and e.target_id != node_id
        ]

        # Remove node
        del self._nodes[node_id]
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": e.relation,
                }
                for e in self._edges
            ],
            "stats": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
            },
        }


@dataclass
class QueryResult:
    """Result from a knowledge query."""

    knowledge: List[Dict[str, Any]]
    memory: Any = None  # MemoryContext from memory layer
