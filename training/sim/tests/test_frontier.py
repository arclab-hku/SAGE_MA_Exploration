"""Tests for frontier detection correctness."""
from __future__ import annotations

import numpy as np
import pytest

from sim.config import GraphConfig, Layer2Config, WorldConfig


def detect_frontiers(
    mask_state: np.ndarray,
    cfg: GraphConfig | None = None,
) -> set[tuple[int, int]]:
    """Standalone frontier detection for testing.
    Frontier: cell is UNKNOWN(-1) AND has at least 1 FREE(0) 4-neighbor.
    Matches production: uav_gogogo_continue_onnx.py:652-735."""
    cfg = cfg or GraphConfig()
    h, w = mask_state.shape
    frontiers: set[tuple[int, int]] = set()
    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if mask_state[r, c] != -1:
                continue
            has_free = False
            for dr, dc in neighbors_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask_state[nr, nc] == 0:
                    has_free = True
                    break
            if has_free:
                frontiers.add((r, c))

    return frontiers


class TestFrontierDetection:

    def test_single_frontier_at_boundary(self):
        """Unknown cell next to free cell is a frontier."""
        grid = np.full((5, 5), -1, dtype=np.int8)
        grid[2, 2] = 0  # one free cell
        frontiers = detect_frontiers(grid)
        # 4-neighbors of (2,2) that are unknown → frontiers
        expected = {(1, 2), (3, 2), (2, 1), (2, 3)}
        assert frontiers == expected

    def test_no_frontier_all_free(self):
        grid = np.zeros((5, 5), dtype=np.int8)
        frontiers = detect_frontiers(grid)
        assert len(frontiers) == 0

    def test_no_frontier_all_unknown(self):
        grid = np.full((5, 5), -1, dtype=np.int8)
        frontiers = detect_frontiers(grid)
        assert len(frontiers) == 0

    def test_occupied_not_frontier(self):
        """Occupied cells next to free cells are NOT frontiers."""
        grid = np.full((5, 5), -1, dtype=np.int8)
        grid[2, 2] = 0
        grid[1, 2] = 100  # occupied neighbor
        frontiers = detect_frontiers(grid)
        assert (1, 2) not in frontiers
        assert (3, 2) in frontiers

    def test_frontier_ring_around_explored(self):
        """Exploring from center should create a ring of frontiers."""
        grid = np.full((20, 20), -1, dtype=np.int8)
        # Free 3x3 area in center
        grid[9:12, 9:12] = 0
        frontiers = detect_frontiers(grid)
        # Frontiers should be the unknown cells adjacent to the free area
        for r, c in frontiers:
            assert grid[r, c] == -1
            # Must have at least one free neighbor
            has_free = any(
                0 <= r + dr < 20 and 0 <= c + dc < 20 and grid[r + dr, c + dc] == 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            assert has_free

    def test_corridor_frontiers(self):
        """In a corridor, frontiers appear at the unexplored ends."""
        grid = np.full((10, 10), -1, dtype=np.int8)
        # Free corridor at row 5, cols 3-6
        grid[5, 3:7] = 0
        frontiers = detect_frontiers(grid)
        # Should include unknown cells adjacent to corridor
        assert (5, 2) in frontiers   # left end
        assert (5, 7) in frontiers   # right end
        assert (4, 3) in frontiers   # above corridor
        assert (6, 3) in frontiers   # below corridor

    def test_frontier_removed_when_explored(self):
        """When an unknown cell becomes free, it's no longer a frontier."""
        grid = np.full((5, 5), -1, dtype=np.int8)
        grid[2, 2] = 0
        f1 = detect_frontiers(grid)
        assert (1, 2) in f1

        grid[1, 2] = 0  # explore the frontier
        f2 = detect_frontiers(grid)
        assert (1, 2) not in f2
        # But new frontiers appear
        assert (0, 2) in f2

    def test_corner_frontier(self):
        """Corner cell with only one free neighbor is still a frontier."""
        grid = np.full((5, 5), -1, dtype=np.int8)
        grid[0, 0] = 0
        frontiers = detect_frontiers(grid)
        assert (0, 1) in frontiers
        assert (1, 0) in frontiers

    def test_large_explored_area(self):
        """Large explored area should have frontiers only at boundary."""
        grid = np.full((20, 20), -1, dtype=np.int8)
        grid[5:15, 5:15] = 0  # 10x10 explored area
        frontiers = detect_frontiers(grid)
        # All frontiers must be outside the free area
        for r, c in frontiers:
            assert grid[r, c] == -1
        # Count: should be around the perimeter of the free area
        assert len(frontiers) > 0
        assert len(frontiers) <= 4 * 10 + 4  # perimeter + corners
