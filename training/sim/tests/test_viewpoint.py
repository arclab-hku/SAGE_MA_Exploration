"""Tests for viewpoint generation + ray cast visibility."""
from __future__ import annotations

import numpy as np
import pytest

from sim.config import GraphConfig, SensorConfig
from sim.graph_state import ExplorationGraph, make_node_id, _count_visible_frontiers


class TestViewpointGeneration:
    """Viewpoints are generated at 4-neighbors of robot position."""

    def _setup_mask_with_frontiers(self):
        """Create mask with explored center and frontier ring."""
        mask = np.full((20, 20), -1, dtype=np.int8)
        mask[8:12, 8:12] = 0  # explored center
        return mask

    def test_viewpoints_at_4_neighbors(self):
        mask = self._setup_mask_with_frontiers()
        graph = ExplorationGraph(source_uav=0)
        gcfg = GraphConfig()
        scfg = SensorConfig(view_range=5, num_rays=72)

        # Detect frontiers first
        graph.detect_frontiers(mask, gcfg)
        assert graph.frontier_count() > 0

        # Generate viewpoints from robot at (10, 10)
        vps = graph.generate_viewpoints(10, 10, mask, scfg, gcfg)
        # Should have viewpoints at some of the 4 neighbors
        vp_positions = {(v.x, v.y) for v in vps}
        possible = {(9, 10), (11, 10), (10, 9), (10, 11)}
        assert vp_positions.issubset(possible)

    def test_viewpoint_not_on_obstacle(self):
        mask = np.full((20, 20), -1, dtype=np.int8)
        mask[8:12, 8:12] = 0
        # Place obstacle at candidate (x=9, y=10) → mask[row=10, col=9]
        mask[10, 9] = 100

        graph = ExplorationGraph(source_uav=0)
        graph.detect_frontiers(mask)
        vps = graph.generate_viewpoints(10, 10, mask)
        vp_positions = {(v.x, v.y) for v in vps}
        assert (9, 10) not in vp_positions  # blocked by obstacle

    def test_viewpoint_edges_created(self):
        mask = self._setup_mask_with_frontiers()
        graph = ExplorationGraph(source_uav=0)
        graph.detect_frontiers(mask)
        graph.generate_viewpoints(10, 10, mask)

        # Should have frontier_viewpoint edges
        fv_edges = [
            e for e in graph.edges.values()
            if e.edge_type == "frontier_viewpoint"
        ]
        assert len(fv_edges) > 0

    def test_no_viewpoint_without_frontiers(self):
        mask = np.zeros((20, 20), dtype=np.int8)  # all free
        graph = ExplorationGraph(source_uav=0)
        graph.detect_frontiers(mask)
        assert graph.frontier_count() == 0

        vps = graph.generate_viewpoints(10, 10, mask)
        assert len(vps) == 0


class TestVisibleFrontierCount:
    """Ray cast correctly counts visible frontiers."""

    def test_unoccluded_frontier_visible(self):
        mask = np.full((20, 20), -1, dtype=np.int8)
        mask[10, 5:15] = 0  # free corridor
        frontiers = {(14, 10)}  # frontier at end of corridor
        count = _count_visible_frontiers(5, 10, frontiers, mask, view_range=15)
        assert count == 1

    def test_occluded_frontier_not_visible(self):
        mask = np.full((20, 20), -1, dtype=np.int8)
        mask[10, 5:15] = 0
        mask[10, 8] = 100  # wall blocks view
        frontiers = {(14, 10)}
        count = _count_visible_frontiers(5, 10, frontiers, mask, view_range=15)
        assert count == 0

    def test_out_of_range_frontier(self):
        mask = np.full((40, 40), -1, dtype=np.int8)
        mask[20, :] = 0
        frontiers = {(35, 20)}
        count = _count_visible_frontiers(5, 20, frontiers, mask, view_range=10)
        assert count == 0  # too far

    def test_multiple_frontiers(self):
        mask = np.full((20, 20), -1, dtype=np.int8)
        mask[8:12, 8:12] = 0
        # Frontiers around the free area
        frontiers = {(7, 10), (12, 10), (10, 7), (10, 12)}
        count = _count_visible_frontiers(10, 10, frontiers, mask, view_range=10)
        assert count == 4  # all visible from center


class TestRobotNode:

    def test_robot_node_update(self):
        graph = ExplorationGraph(source_uav=0)
        node = graph.update_robot_node(0, 10, 10)
        assert node.node_type == GraphConfig.ROBOT_SELF
        assert node.x == 10 and node.y == 10

    def test_robot_node_moves(self):
        graph = ExplorationGraph(source_uav=0)
        graph.update_robot_node(0, 10, 10)
        graph.update_robot_node(0, 15, 15)
        # Old position should be deleted, only new exists
        robot_nodes = [
            n for n in graph.nodes.values()
            if n.node_type == GraphConfig.ROBOT_SELF and not n.deleted
        ]
        assert len(robot_nodes) == 1
        assert robot_nodes[0].x == 15 and robot_nodes[0].y == 15
