"""
Two-level planner: Layer2 A* (long distance) + Layer1 fine (short distance).

Adjustments:
  A4: Thresholds in meters, converted to cells via resolution
  A5: Fallback = BFS toward nearest unknown, not random walk
  A6: NavEdge budget-limited pruning handled in graph_state.py
"""
from __future__ import annotations

import heapq
from typing import List, Optional, Set, Tuple

import numpy as np

from .config import PlannerConfig, WorldConfig
from .graph_state import ExplorationGraph, GraphNode


# ---------------------------------------------------------------------------
# A* on 2D grid (matches production astar.py pattern but with explicit goal)
# ---------------------------------------------------------------------------

def astar_to_goal(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    unknown_cost: float = 1.0,
) -> Optional[List[Tuple[int, int]]]:
    """A* from start to goal on 2D grid. Returns path or None.
    grid values: -1=unknown, 0=free, 100=occupied.
    Moves through free and unknown cells; obstacles blocked.
    unknown_cost: traversal cost for unknown cells (>1.0 to prefer known-free paths)."""
    h, w = grid.shape
    sr, sc = start
    gr, gc = goal

    if grid[gr, gc] == 100 or grid[sr, sc] == 100:
        return None

    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(r: int, c: int) -> float:
        return abs(r - gr) + abs(c - gc)

    open_set: list = [(heuristic(sr, sc), 0.0, sr, sc)]
    g_cost = {(sr, sc): 0.0}
    came_from = {(sr, sc): None}

    while open_set:
        _, cost, r, c = heapq.heappop(open_set)

        if r == gr and c == gc:
            path = []
            pos = (r, c)
            while pos is not None:
                path.append(pos)
                pos = came_from[pos]
            return path[::-1]

        if cost > g_cost.get((r, c), float("inf")):
            continue

        for dr, dc in movements:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 100:
                step_cost = unknown_cost if grid[nr, nc] == -1 else 1.0
                new_cost = cost + step_cost
                if new_cost < g_cost.get((nr, nc), float("inf")):
                    g_cost[(nr, nc)] = new_cost
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(
                        open_set, (new_cost + heuristic(nr, nc), new_cost, nr, nc)
                    )

    return None


def bfs_nearest_unknown(
    grid: np.ndarray,
    start: Tuple[int, int],
    max_radius: int = 20,
) -> Optional[Tuple[int, int]]:
    """BFS to find nearest unknown cell reachable through free cells (A5)."""
    h, w = grid.shape
    sr, sc = start
    visited: Set[Tuple[int, int]] = {(sr, sc)}
    queue = [(sr, sc, 0)]
    qi = 0
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while qi < len(queue):
        r, c, dist = queue[qi]
        qi += 1
        if dist > max_radius:
            break
        if grid[r, c] == -1 and dist > 0:
            return (r, c)
        for dr, dc in movements:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid[nr, nc] != 100:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
    return None


# ---------------------------------------------------------------------------
# Frontier Planner
# ---------------------------------------------------------------------------

class FrontierPlanner:
    """Greedy frontier selection with two-level A* planning."""

    def __init__(self, pcfg: PlannerConfig, wcfg: WorldConfig):
        self._pcfg = pcfg
        self._wcfg = wcfg
        self._res = wcfg.resolution
        self._current_layer: str = "layer2"  # start with coarse planning
        self._replan_count: int = 0

    @property
    def replan_count(self) -> int:
        return self._replan_count

    def reset_replan_count(self) -> None:
        self._replan_count = 0

    def select_frontier(
        self,
        robot_pos: Tuple[int, int],
        graph: ExplorationGraph,
        mask_state: np.ndarray,
    ) -> Optional[GraphNode]:
        """Score and select best frontier. Returns None if no frontiers."""
        frontiers = graph.get_frontier_nodes()
        if not frontiers:
            return None

        best_score = -float("inf")
        best_node = None

        for f in frontiers:
            dist = abs(f.x - robot_pos[1]) + abs(f.y - robot_pos[0])
            utility = graph.compute_utility(f.node_id)
            score = (
                self._pcfg.utility_weight * utility
                - self._pcfg.distance_weight * dist
            )
            if score > best_score:
                best_score = score
                best_node = f

        return best_node

    def plan_path(
        self,
        robot_pos: Tuple[int, int],
        target: GraphNode,
        layer1_grid: np.ndarray,
        layer2_grid: np.ndarray,
    ) -> Optional[List[Tuple[int, int]]]:
        """Plan path using two-level strategy (A4 thresholds in meters)."""
        goal = (target.y, target.x)  # (row, col)
        dist = abs(target.x - robot_pos[1]) + abs(target.y - robot_pos[0])

        l2_thresh = self._pcfg.layer2_switch_cells(self._res)
        l1_thresh = self._pcfg.layer1_switch_cells(self._res)
        hyst = self._pcfg.hysteresis_cells(self._res)

        # Layer switching with hysteresis (A4)
        if self._current_layer == "layer2":
            if dist < l1_thresh:
                self._current_layer = "layer1"
        else:
            if dist > l2_thresh + hyst:
                self._current_layer = "layer2"

        if self._current_layer == "layer2":
            path = astar_to_goal(layer2_grid, robot_pos, goal)
        else:
            path = astar_to_goal(layer1_grid, robot_pos, goal)

        if path is not None:
            return path

        # Fallback: try the other layer
        self._replan_count += 1
        if self._replan_count > self._pcfg.max_replan_per_step:
            return None

        alt_grid = layer1_grid if self._current_layer == "layer2" else layer2_grid
        path = astar_to_goal(alt_grid, robot_pos, goal)
        if path is not None:
            return path

        return None

    def fallback_step(
        self,
        robot_pos: Tuple[int, int],
        mask_state: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Fallback: BFS toward nearest unknown (A5)."""
        target = bfs_nearest_unknown(
            mask_state, robot_pos, self._pcfg.fallback_bfs_radius,
        )
        if target is None:
            return None

        path = astar_to_goal(mask_state, robot_pos, target)
        if path and len(path) > 1:
            return path[1]
        return None
