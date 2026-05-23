"""
CRDT resolver + GraphDelta message for Layer 3 (exploration graph).

Rules:
  - node_id = quantized_coord + type (spatial hash dedup)
  - Version = (lamport_counter, source_uav) — A1
  - DELETE wins at same version (tombstone + TTL)
  - NavEdge conflict: blocked=True wins (safety bias)
  - Utility never synced, locally derived from edge count
  - Tombstone GC after TTL steps
"""
from __future__ import annotations

from dataclasses import dataclass

from .graph_state import ExplorationGraph, GraphEdge, GraphNode, VersionT, _ZERO_VER


# ---------------------------------------------------------------------------
# GraphDelta message
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeDelta:
    node_id: str
    node_type: int
    x: int
    y: int
    version: VersionT
    deleted: bool


@dataclass(frozen=True)
class EdgeDelta:
    src_id: str
    dst_id: str
    edge_type: str
    blocked: bool
    version: VersionT


@dataclass(frozen=True)
class GraphDelta:
    """Incremental graph sync message."""
    source_uav: int
    seq: int
    node_deltas: tuple[NodeDelta, ...]
    edge_deltas: tuple[EdgeDelta, ...]

    @property
    def byte_size(self) -> int:
        # 8 header + 16/node + 8/edge
        return 8 + 16 * len(self.node_deltas) + 8 * len(self.edge_deltas)


# ---------------------------------------------------------------------------
# Delta production
# ---------------------------------------------------------------------------

class DeltaTracker:
    """Tracks changes to produce GraphDelta messages."""

    def __init__(self, source_uav: int):
        self._source = source_uav
        self._seq = 0
        self._pending_nodes: list[NodeDelta] = []
        self._pending_edges: list[EdgeDelta] = []

    def record_node(self, node: GraphNode) -> None:
        self._pending_nodes.append(NodeDelta(
            node_id=node.node_id,
            node_type=node.node_type,
            x=node.x, y=node.y,
            version=node.version,
            deleted=node.deleted,
        ))

    def record_edge(self, edge: GraphEdge) -> None:
        self._pending_edges.append(EdgeDelta(
            src_id=edge.src_id,
            dst_id=edge.dst_id,
            edge_type=edge.edge_type,
            blocked=edge.blocked,
            version=edge.version,
        ))

    def flush(self) -> GraphDelta | None:
        if not self._pending_nodes and not self._pending_edges:
            return None
        self._seq += 1
        delta = GraphDelta(
            source_uav=self._source,
            seq=self._seq,
            node_deltas=tuple(self._pending_nodes),
            edge_deltas=tuple(self._pending_edges),
        )
        self._pending_nodes.clear()
        self._pending_edges.clear()
        return delta


# ---------------------------------------------------------------------------
# CRDT Resolver
# ---------------------------------------------------------------------------

class CRDTResolver:
    """Merges remote GraphDelta into a local ExplorationGraph."""

    def merge(
        self,
        graph: ExplorationGraph,
        delta: GraphDelta,
        current_step: int = 0,
    ) -> None:
        """Apply delta to graph using CRDT rules."""
        # Advance Lamport clock
        for nd in delta.node_deltas:
            graph.advance_lamport(nd.version[0])
        for ed in delta.edge_deltas:
            graph.advance_lamport(ed.version[0])

        for nd in delta.node_deltas:
            self._merge_node(graph, nd, current_step)

        for ed in delta.edge_deltas:
            self._merge_edge(graph, ed)

    def _merge_node(
        self,
        graph: ExplorationGraph,
        nd: NodeDelta,
        current_step: int,
    ) -> None:
        existing = graph.nodes.get(nd.node_id)

        if existing is None:
            # New node — check tombstone
            if nd.node_id in graph.tombstones and not nd.deleted:
                # Tombstone exists: only accept if version > tombstone version
                # (scenario 3: old ADD after DELETE)
                tombstone_node = graph.nodes.get(nd.node_id)
                if tombstone_node and nd.version <= tombstone_node.version:
                    return
            # Accept new node
            node = GraphNode(
                node_id=nd.node_id,
                node_type=nd.node_type,
                x=nd.x, y=nd.y,
                version=nd.version,
                deleted=nd.deleted,
            )
            graph.nodes[nd.node_id] = node
            if nd.deleted:
                graph.tombstones[nd.node_id] = current_step
            return

        # Existing node — version comparison
        if nd.version > existing.version:
            # Remote is newer: accept
            node = GraphNode(
                node_id=nd.node_id,
                node_type=nd.node_type,
                x=nd.x, y=nd.y,
                version=nd.version,
                deleted=nd.deleted,
            )
            graph.nodes[nd.node_id] = node
            if nd.deleted:
                graph.tombstones[nd.node_id] = current_step
                # Clean up edges
                to_remove = [
                    eid for eid in graph.edges
                    if eid[0] == nd.node_id or eid[1] == nd.node_id
                ]
                for eid in to_remove:
                    del graph.edges[eid]
        elif nd.version == existing.version:
            # Same version: DELETE wins (CRDT rule)
            if nd.deleted and not existing.deleted:
                node = GraphNode(
                    node_id=nd.node_id,
                    node_type=nd.node_type,
                    x=nd.x, y=nd.y,
                    version=nd.version,
                    deleted=True,
                )
                graph.nodes[nd.node_id] = node
                graph.tombstones[nd.node_id] = current_step
                to_remove = [
                    eid for eid in graph.edges
                    if eid[0] == nd.node_id or eid[1] == nd.node_id
                ]
                for eid in to_remove:
                    del graph.edges[eid]
        # else: local is newer, ignore

    def _merge_edge(self, graph: ExplorationGraph, ed: EdgeDelta) -> None:
        eid = (ed.src_id, ed.dst_id, ed.edge_type)
        existing = graph.edges.get(eid)

        if existing is None:
            # New edge
            graph.edges[eid] = GraphEdge(
                src_id=ed.src_id, dst_id=ed.dst_id,
                edge_type=ed.edge_type,
                blocked=ed.blocked,
                version=ed.version,
            )
            return

        if ed.version > existing.version:
            # Remote is newer: accept
            graph.edges[eid] = GraphEdge(
                src_id=ed.src_id, dst_id=ed.dst_id,
                edge_type=ed.edge_type,
                blocked=ed.blocked,
                version=ed.version,
            )
        elif ed.version == existing.version:
            # Same version NavEdge conflict: blocked=True wins (safety bias)
            if ed.edge_type == "nav" and ed.blocked and not existing.blocked:
                graph.edges[eid] = GraphEdge(
                    src_id=ed.src_id, dst_id=ed.dst_id,
                    edge_type=ed.edge_type,
                    blocked=True,
                    version=ed.version,
                )
