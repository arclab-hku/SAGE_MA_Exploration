"""Tests for CRDT conflict resolution — all 7 scenarios."""
from __future__ import annotations

import pytest

from sim.crdt import CRDTResolver, EdgeDelta, GraphDelta, NodeDelta
from sim.graph_state import ExplorationGraph, GraphEdge, GraphNode, make_node_id


def _make_graph(uav: int = 0) -> ExplorationGraph:
    return ExplorationGraph(source_uav=uav)


def _make_node_delta(
    nid: str, ntype: int, x: int, y: int,
    version: tuple[int, int], deleted: bool = False,
) -> NodeDelta:
    return NodeDelta(
        node_id=nid, node_type=ntype, x=x, y=y,
        version=version, deleted=deleted,
    )


def _make_graph_delta(
    source: int,
    nodes: list[NodeDelta] | None = None,
    edges: list[EdgeDelta] | None = None,
) -> GraphDelta:
    return GraphDelta(
        source_uav=source, seq=1,
        node_deltas=tuple(nodes or []),
        edge_deltas=tuple(edges or []),
    )


class TestCRDTScenario1ConcurrentAddSameFrontier:
    """Two UAVs both discover the same frontier → deduplicated via spatial hash."""

    def test_dedup_same_frontier(self):
        graph = _make_graph(0)
        resolver = CRDTResolver()
        nid = make_node_id(0, 10, 20)

        # UAV 0 adds frontier
        delta_0 = _make_graph_delta(0, [
            _make_node_delta(nid, 0, 10, 20, (1, 0)),
        ])
        resolver.merge(graph, delta_0)
        assert nid in graph.nodes
        assert not graph.nodes[nid].deleted

        # UAV 1 adds same frontier (higher version)
        delta_1 = _make_graph_delta(1, [
            _make_node_delta(nid, 0, 10, 20, (2, 1)),
        ])
        resolver.merge(graph, delta_1)
        # Still one node, updated version
        assert graph.nodes[nid].version == (2, 1)

    def test_dedup_same_version_different_source(self):
        graph = _make_graph(0)
        resolver = CRDTResolver()
        nid = make_node_id(0, 10, 20)

        delta_0 = _make_graph_delta(0, [
            _make_node_delta(nid, 0, 10, 20, (1, 0)),
        ])
        resolver.merge(graph, delta_0)

        # Same lamport, different source — (1, 1) > (1, 0) by source tie-break
        delta_1 = _make_graph_delta(1, [
            _make_node_delta(nid, 0, 10, 20, (1, 1)),
        ])
        resolver.merge(graph, delta_1)
        assert graph.nodes[nid].version == (1, 1)


class TestCRDTScenario2DeleteWinsOverAdd:
    """DELETE at same version beats ADD → tombstone blocks stale ADD."""

    def test_delete_wins_same_version(self):
        graph = _make_graph(0)
        resolver = CRDTResolver()
        nid = make_node_id(0, 5, 5)

        # First add the node
        delta_add = _make_graph_delta(0, [
            _make_node_delta(nid, 0, 5, 5, (1, 0)),
        ])
        resolver.merge(graph, delta_add)
        assert not graph.nodes[nid].deleted

        # DELETE at same version
        delta_del = _make_graph_delta(1, [
            _make_node_delta(nid, 0, 5, 5, (1, 0), deleted=True),
        ])
        resolver.merge(graph, delta_del)
        assert graph.nodes[nid].deleted


class TestCRDTScenario3OldAddAfterDelete:
    """Old ADD arriving after DELETE is rejected by version check."""

    def test_old_add_rejected(self):
        graph = _make_graph(0)
        resolver = CRDTResolver()
        nid = make_node_id(0, 7, 7)

        # Delete at version (3, 0)
        delta_del = _make_graph_delta(0, [
            _make_node_delta(nid, 0, 7, 7, (3, 0), deleted=True),
        ])
        resolver.merge(graph, delta_del)
        assert graph.nodes[nid].deleted

        # Stale ADD at version (1, 1) — should be ignored
        delta_old = _make_graph_delta(1, [
            _make_node_delta(nid, 0, 7, 7, (1, 1)),
        ])
        resolver.merge(graph, delta_old)
        assert graph.nodes[nid].deleted  # still deleted


class TestCRDTScenario4FrontierReappears:
    """Higher version ADD revives a deleted node (dynamic environment)."""

    def test_revive_with_higher_version(self):
        graph = _make_graph(0)
        resolver = CRDTResolver()
        nid = make_node_id(0, 12, 12)

        # Delete at (2, 0)
        delta_del = _make_graph_delta(0, [
            _make_node_delta(nid, 0, 12, 12, (2, 0), deleted=True),
        ])
        resolver.merge(graph, delta_del)
        assert graph.nodes[nid].deleted

        # Re-ADD at higher version (5, 1)
        delta_revive = _make_graph_delta(1, [
            _make_node_delta(nid, 0, 12, 12, (5, 1)),
        ])
        resolver.merge(graph, delta_revive)
        assert not graph.nodes[nid].deleted
        assert graph.nodes[nid].version == (5, 1)


class TestCRDTScenario5UtilityDisagreement:
    """Utility is derived locally from edge count, not synced."""

    def test_utility_from_edges(self):
        graph = _make_graph(0)
        fid = make_node_id(0, 10, 10)
        vid1 = make_node_id(1, 9, 10)
        vid2 = make_node_id(1, 11, 10)

        graph.nodes[fid] = GraphNode(fid, 0, 10, 10, (1, 0))
        graph.nodes[vid1] = GraphNode(vid1, 1, 9, 10, (1, 0))
        graph.nodes[vid2] = GraphNode(vid2, 1, 11, 10, (1, 0))

        # Two viewpoints can see this frontier
        graph.edges[(vid1, fid, "frontier_viewpoint")] = GraphEdge(
            vid1, fid, "frontier_viewpoint", version=(1, 0),
        )
        graph.edges[(vid2, fid, "frontier_viewpoint")] = GraphEdge(
            vid2, fid, "frontier_viewpoint", version=(1, 0),
        )

        assert graph.compute_utility(fid) == 2

        # Even if remote says utility=5, local re-derives from edges
        # (no utility field in delta, so nothing to merge)


class TestCRDTScenario6NavEdgeContradiction:
    """NavEdge conflict at same version: blocked=True wins (safety bias)."""

    def test_blocked_wins(self):
        graph = _make_graph(0)
        resolver = CRDTResolver()

        # UAV 0 says edge is free
        graph.edges[("nav_1_1", "nav_5_5", "nav")] = GraphEdge(
            "nav_1_1", "nav_5_5", "nav", blocked=False, version=(3, 0),
        )

        # UAV 1 says edge is blocked at same version
        delta = _make_graph_delta(1, edges=[
            EdgeDelta("nav_1_1", "nav_5_5", "nav", blocked=True, version=(3, 0)),
        ])
        resolver.merge(graph, delta)
        edge = graph.edges[("nav_1_1", "nav_5_5", "nav")]
        assert edge.blocked is True

    def test_newer_version_overrides(self):
        graph = _make_graph(0)
        resolver = CRDTResolver()

        graph.edges[("nav_1_1", "nav_5_5", "nav")] = GraphEdge(
            "nav_1_1", "nav_5_5", "nav", blocked=True, version=(2, 0),
        )

        # Newer version says free
        delta = _make_graph_delta(1, edges=[
            EdgeDelta("nav_1_1", "nav_5_5", "nav", blocked=False, version=(4, 1)),
        ])
        resolver.merge(graph, delta)
        assert graph.edges[("nav_1_1", "nav_5_5", "nav")].blocked is False


class TestCRDTScenario7NetworkPartitionReconnect:
    """After partition, version vector diff enables incremental sync."""

    def test_partition_and_reconnect(self):
        graph_a = _make_graph(0)
        graph_b = _make_graph(1)
        resolver = CRDTResolver()

        # During partition: A discovers frontiers
        nid1 = make_node_id(0, 3, 3)
        nid2 = make_node_id(0, 4, 4)
        graph_a.nodes[nid1] = GraphNode(nid1, 0, 3, 3, (1, 0))
        graph_a.nodes[nid2] = GraphNode(nid2, 0, 4, 4, (2, 0))

        # During partition: B discovers different frontiers
        nid3 = make_node_id(0, 30, 30)
        graph_b.nodes[nid3] = GraphNode(nid3, 0, 30, 30, (1, 1))

        # Reconnect: A sends its state to B
        delta_a = GraphDelta(
            source_uav=0, seq=1,
            node_deltas=tuple(
                NodeDelta(n.node_id, n.node_type, n.x, n.y, n.version, n.deleted)
                for n in graph_a.nodes.values()
            ),
            edge_deltas=(),
        )
        resolver.merge(graph_b, delta_a)

        # B should now have all 3 nodes
        assert nid1 in graph_b.nodes
        assert nid2 in graph_b.nodes
        assert nid3 in graph_b.nodes

        # B sends its state to A
        delta_b = GraphDelta(
            source_uav=1, seq=1,
            node_deltas=tuple(
                NodeDelta(n.node_id, n.node_type, n.x, n.y, n.version, n.deleted)
                for n in graph_b.nodes.values()
            ),
            edge_deltas=(),
        )
        resolver.merge(graph_a, delta_b)

        # A should now also have all 3 nodes
        assert nid1 in graph_a.nodes
        assert nid2 in graph_a.nodes
        assert nid3 in graph_a.nodes

    def test_converge_after_partition(self):
        """Both graphs converge to identical state after bidirectional sync."""
        graph_a = _make_graph(0)
        graph_b = _make_graph(1)
        resolver = CRDTResolver()

        nid = make_node_id(0, 10, 10)

        # A has old version
        graph_a.nodes[nid] = GraphNode(nid, 0, 10, 10, (1, 0))
        # B deleted it at higher version
        graph_b.nodes[nid] = GraphNode(nid, 0, 10, 10, (3, 1), deleted=True)
        graph_b.tombstones[nid] = 0

        # Bidirectional sync
        delta_a = _make_graph_delta(0, [
            _make_node_delta(nid, 0, 10, 10, (1, 0)),
        ])
        delta_b = _make_graph_delta(1, [
            _make_node_delta(nid, 0, 10, 10, (3, 1), deleted=True),
        ])

        resolver.merge(graph_b, delta_a)
        resolver.merge(graph_a, delta_b)

        # Both should agree: deleted at (3, 1)
        assert graph_a.nodes[nid].deleted
        assert graph_b.nodes[nid].deleted
        assert graph_a.nodes[nid].version == (3, 1)
        assert graph_b.nodes[nid].version == (3, 1)
