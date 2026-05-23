"""
UAV agent: composes sensor + Layer2 mask + Layer3 graph + planner.

Fixes applied:
  - Stale-aware frontier scoring: penalize frontiers with old information age
  - Corridor commit: in narrow passages, commit to path without re-evaluation
  - Collision split: static obstacle vs inter-UAV (separate counters)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import (
    GraphConfig,
    Layer2Config,
    PlannerConfig,
    SensorConfig,
    WorldConfig,
)
from .crdt import CRDTResolver, DeltaTracker, GraphDelta
from .explored_mask import ExploredMask, Layer2Delta
from .graph_state import ExplorationGraph
from .planner import FrontierPlanner, astar_to_goal, bfs_nearest_unknown
from .sensor import cast_rays


class UAVAgent:
    """Single UAV agent with 3-layer architecture."""

    def __init__(
        self,
        uav_id: int,
        start_pos: Tuple[int, int],
        wcfg: WorldConfig,
        scfg: SensorConfig,
        l2cfg: Layer2Config,
        gcfg: GraphConfig,
        pcfg: PlannerConfig,
    ):
        self.uav_id = uav_id
        self._pos: Tuple[int, int] = start_pos
        self._wcfg = wcfg
        self._scfg = scfg
        self._gcfg = gcfg
        self._pcfg = pcfg

        # Layer 2: explored mask
        self.mask = ExploredMask(wcfg, source_uav=uav_id, l2_cfg=l2cfg)
        # Layer 3: exploration graph
        self.graph = ExplorationGraph(source_uav=uav_id)
        self._delta_tracker = DeltaTracker(source_uav=uav_id)
        self._crdt = CRDTResolver()
        # Planner
        self._planner = FrontierPlanner(pcfg, wcfg)
        # Current path + corridor commit
        self._path: Optional[List[Tuple[int, int]]] = None
        self._path_idx: int = 0
        self._committed: bool = False  # corridor commit flag
        self._steps_on_path: int = 0

        # Information age tracking: step when each cell was last observed
        self._last_observed = np.full(
            (wcfg.height, wcfg.width), -1, dtype=np.int32,
        )

        # Stats — split collisions
        self.collisions_static: int = 0    # hit wall/obstacle (Layer2 delay)
        self.collisions_inter_uav: int = 0  # hit another UAV position
        self.replans: int = 0
        self.frontier_expirations: int = 0

    @property
    def pos(self) -> Tuple[int, int]:
        return self._pos

    @property
    def collisions(self) -> int:
        return self.collisions_static + self.collisions_inter_uav

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        ground_truth: np.ndarray,
        current_step: int,
        all_uav_ids: List[int],
        other_positions: Dict[int, Tuple[int, int]],
    ) -> Tuple[Optional[Layer2Delta], Optional[GraphDelta]]:
        # 1. Sense
        free_cells, occ_cells = cast_rays(
            self._pos[1], self._pos[0], ground_truth, self._scfg,
        )

        # 2. Update Layer2 mask + information age
        self.mask.update_from_sensor(free_cells, occ_cells)
        for r, c in free_cells:
            self._last_observed[r, c] = current_step
        for r, c in occ_cells:
            self._last_observed[r, c] = current_step

        if current_step > 0 and current_step % 50 == 0:
            self.mask.decay_tick()

        # 3. Rebuild Layer3 graph
        old_frontier_ids = {n.node_id for n in self.graph.get_frontier_nodes()}
        new_frontier_nodes = self.graph.detect_frontiers(self.mask.state, self._gcfg)
        current_frontier_ids = {n.node_id for n in self.graph.get_frontier_nodes()}
        removed_ids = old_frontier_ids - current_frontier_ids
        self.frontier_expirations += len(removed_ids)

        self.graph.generate_viewpoints(
            self._pos[1], self._pos[0], self.mask.state, self._scfg, self._gcfg,
        )
        self.graph.update_robot_node(
            self.uav_id, self._pos[1], self._pos[0], is_self=True, gcfg=self._gcfg,
        )
        for uid, pos in other_positions.items():
            if uid != self.uav_id:
                self.graph.update_robot_node(
                    uid, pos[1], pos[0], is_self=False, gcfg=self._gcfg,
                )

        self.graph.prune_nav_edges(
            self._pos[1], self._pos[0], self.mask.state,
            self._scfg.view_range, self._gcfg,
        )
        self.graph.gc_tombstones(current_step)

        # 4. Flush deltas
        layer2_delta = self.mask.flush_delta()
        for node in new_frontier_nodes:
            self._delta_tracker.record_node(node)
        for rid in removed_ids:
            rnode = self.graph.nodes.get(rid)
            if rnode:
                self._delta_tracker.record_node(rnode)
        graph_delta = self._delta_tracker.flush()

        # 5. Plan & move
        self._planner.reset_replan_count()
        self._plan_and_move(ground_truth, current_step, other_positions)
        self.replans += self._planner.replan_count

        return layer2_delta, graph_delta

    # ------------------------------------------------------------------
    # Planning with stale-aware scoring + corridor commit
    # ------------------------------------------------------------------

    def _plan_and_move(
        self,
        ground_truth: np.ndarray,
        current_step: int,
        other_positions: Dict[int, Tuple[int, int]],
    ) -> None:
        # Corridor commit: if committed, keep following path
        if self._committed and self._path and self._path_idx < len(self._path):
            next_pos = self._path[self._path_idx]
            if self._try_move(next_pos, ground_truth, other_positions):
                self._path_idx += 1
                self._steps_on_path += 1
                # Release commitment if we exit narrow passage
                if not self._is_in_corridor():
                    self._committed = False
                return
            else:
                # Path blocked even in corridor — must replan
                self._path = None
                self._committed = False

        # If we have a valid path (not committed), follow but allow re-eval
        if self._path and self._path_idx < len(self._path):
            next_pos = self._path[self._path_idx]
            if self._try_move(next_pos, ground_truth, other_positions):
                self._path_idx += 1
                self._steps_on_path += 1
                # Enter corridor commit if in narrow passage
                if self._is_in_corridor():
                    self._committed = True
                return
            else:
                self._path = None

        # Select nearest frontier (same logic as baseline for fair A7 comparison)
        target = self._select_nearest_frontier()
        if target is not None:
            path = astar_to_goal(
                self.mask.state, self._pos, target,
                unknown_cost=self._pcfg.unknown_cell_cost,
            )
            if path and len(path) > 1:
                self._path = path
                self._path_idx = 1
                self._steps_on_path = 0
                if self._try_move(path[1], ground_truth, other_positions):
                    self._steps_on_path = 1
                    if self._is_in_corridor():
                        self._committed = True
                return

        # Fallback
        fallback = self._planner.fallback_step(self._pos, self.mask.state)
        if fallback:
            self._try_move(fallback, ground_truth, other_positions)

    def _select_nearest_frontier(self) -> Optional[Tuple[int, int]]:
        """Select nearest frontier (same logic as baseline for fair A7 comparison).
        Vectorized: unknown cell with at least one free 4-neighbor."""
        m = self.mask.state
        is_unknown = m == -1
        is_free = m == 0
        padded = np.pad(is_free, 1, mode="constant", constant_values=False)
        has_free_neighbor = (
            padded[:-2, 1:-1] | padded[2:, 1:-1]
            | padded[1:-1, :-2] | padded[1:-1, 2:]
        )
        frontier_mask = is_unknown & has_free_neighbor
        coords = np.argwhere(frontier_mask)
        if len(coords) == 0:
            return None

        dists = np.abs(coords[:, 0] - self._pos[0]) + np.abs(coords[:, 1] - self._pos[1])
        best_idx = np.argmin(dists)
        return (int(coords[best_idx, 0]), int(coords[best_idx, 1]))

    def _is_in_corridor(self) -> bool:
        """Check if current position is in a narrow passage.
        Uses configurable threshold (default 1 = only dead-end-like cells)."""
        r, c = self._pos
        m = self.mask.state
        h, w = m.shape
        free_count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and m[nr, nc] == 0:
                free_count += 1
        return free_count <= self._pcfg.corridor_max_free_neighbors

    def _try_move(
        self,
        target: Tuple[int, int],
        ground_truth: np.ndarray,
        other_positions: Dict[int, Tuple[int, int]],
    ) -> bool:
        """Move to adjacent cell. Split collision tracking.
        Inter-UAV collisions are tracked but don't block movement
        (matches baseline behavior for fair A7 comparison)."""
        r, c = target
        h, w = ground_truth.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False

        # Static obstacle collision — blocks movement
        if ground_truth[r, c] == 100:
            self.collisions_static += 1
            return False

        # Inter-UAV proximity tracking (non-blocking, for metrics only)
        for uid, opos in other_positions.items():
            if uid != self.uav_id and opos == (r, c):
                self.collisions_inter_uav += 1

        self._pos = (r, c)
        return True

    # ------------------------------------------------------------------
    # Receive remote deltas
    # ------------------------------------------------------------------

    def receive_layer2_delta(self, delta: Layer2Delta) -> None:
        self.mask.merge_delta(delta)

    def receive_graph_delta(self, delta: GraphDelta, step: int = 0) -> None:
        self._crdt.merge(self.graph, delta, step)
