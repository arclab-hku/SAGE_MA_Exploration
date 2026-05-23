"""
GBPlanner UAV agent: RRG-based exploration with greedy gain/cost selection.

Reuses existing modules:
  - ExploredMask for Layer2 occupancy grid
  - cast_rays for sensing
  - astar_to_goal for grid-level path planning
  - bfs_nearest_unknown for fallback
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    GBPlannerConfig,
    Layer2Config,
    PlannerConfig,
    SensorConfig,
    WorldConfig,
)
from .explored_mask import ExploredMask, Layer2Delta
from .gbplanner import build_rrg, select_best_vertex, voronoi_partition
from .planner import astar_to_goal, bfs_nearest_unknown
from .sensor import cast_rays


class GBPlannerUAV:
    """UAV using GBPlanner-style RRG + greedy gain/cost selection."""

    def __init__(
        self,
        uav_id: int,
        start_pos: Tuple[int, int],
        wcfg: WorldConfig,
        scfg: SensorConfig,
        l2cfg: Layer2Config,
        pcfg: PlannerConfig,
        gbcfg: GBPlannerConfig,
    ):
        self.uav_id = uav_id
        self._pos: Tuple[int, int] = start_pos
        self._wcfg = wcfg
        self._scfg = scfg
        self._pcfg = pcfg
        self._gbcfg = gbcfg

        # Layer2 occupancy grid (same as proposed system)
        self.mask = ExploredMask(wcfg, source_uav=uav_id, l2_cfg=l2cfg)

        # RRG state
        self._rng = np.random.RandomState(42 + uav_id)
        self._path: Optional[List[Tuple[int, int]]] = None
        self._path_idx: int = 0
        self._steps_since_rebuild: int = 0
        self._target: Optional[Tuple[int, int]] = None

        # Stats
        self.collisions_static: int = 0
        self.collisions_inter_uav: int = 0
        self.replans: int = 0
        self.frontier_expirations: int = 0

    @property
    def pos(self) -> Tuple[int, int]:
        return self._pos

    @property
    def collisions(self) -> int:
        return self.collisions_static + self.collisions_inter_uav

    def step(
        self,
        ground_truth: np.ndarray,
        current_step: int,
        all_positions: Dict[int, Tuple[int, int]],
    ) -> Optional[Layer2Delta]:
        """Run one step: sense, plan via RRG, move.

        Returns Layer2Delta for communication (or None).
        """
        other_positions = {k: v for k, v in all_positions.items() if k != self.uav_id}

        # 1. Sense
        free_cells, occ_cells = cast_rays(
            self._pos[1], self._pos[0], ground_truth, self._scfg,
        )

        # 2. Update Layer2
        self.mask.update_from_sensor(free_cells, occ_cells)
        if current_step > 0 and current_step % 50 == 0:
            self.mask.decay_tick()

        # 3. Plan & Move
        self._plan_and_move(ground_truth, current_step, all_positions, other_positions)

        # 4. Flush Layer2 delta
        return self.mask.flush_delta()

    def _plan_and_move(
        self,
        ground_truth: np.ndarray,
        current_step: int,
        all_positions: Dict[int, Tuple[int, int]],
        other_positions: Dict[int, Tuple[int, int]],
    ) -> None:
        # Follow existing path if valid
        if self._path and self._path_idx < len(self._path):
            next_pos = self._path[self._path_idx]
            if self._try_move(next_pos, ground_truth, other_positions):
                self._path_idx += 1
                self._steps_since_rebuild += 1
                # Check if we need RRG rebuild
                if self._steps_since_rebuild < self._gbcfg.rebuild_interval:
                    return
            else:
                # Path blocked, force replan
                self._path = None
                self.replans += 1

        # RRG rebuild + target selection
        self._steps_since_rebuild = 0

        # Voronoi partition
        voronoi = voronoi_partition(all_positions, self.mask.state)
        my_voronoi = voronoi.get(self.uav_id)

        # Build RRG in our partition
        rrg = build_rrg(
            self.mask.state, self._pos, self._rng,
            self._gbcfg, self._scfg, voronoi_mask=my_voronoi,
        )

        # Select best vertex (gain/cost greedy)
        # Robot node is always ID 0 (first added)
        best = select_best_vertex(rrg, 0, self._gbcfg.min_gain_threshold)

        if best is not None:
            target = (best.y, best.x)  # (row, col)
            path = astar_to_goal(
                self.mask.state, self._pos, target,
                unknown_cost=self._pcfg.unknown_cell_cost,
            )
            if path and len(path) > 1:
                self._path = path
                self._path_idx = 1
                self._target = target
                self._try_move(path[1], ground_truth, other_positions)
                return

        # Fallback: nearest frontier (same as baseline/proposed)
        target = self._select_nearest_frontier()
        if target is not None:
            path = astar_to_goal(
                self.mask.state, self._pos, target,
                unknown_cost=self._pcfg.unknown_cell_cost,
            )
            if path and len(path) > 1:
                self._path = path
                self._path_idx = 1
                self._target = target
                self._try_move(path[1], ground_truth, other_positions)
                return

        # Last-resort fallback: BFS toward unknown
        fallback_target = bfs_nearest_unknown(
            self.mask.state, self._pos, self._pcfg.fallback_bfs_radius,
        )
        if fallback_target is not None:
            path = astar_to_goal(self.mask.state, self._pos, fallback_target)
            if path and len(path) > 1:
                self._path = path
                self._path_idx = 1
                self._try_move(path[1], ground_truth, other_positions)

    def _select_nearest_frontier(self) -> Optional[Tuple[int, int]]:
        """Vectorized nearest-frontier selection (same as baseline/proposed)."""
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

    def _try_move(
        self,
        target: Tuple[int, int],
        ground_truth: np.ndarray,
        other_positions: Dict[int, Tuple[int, int]],
    ) -> bool:
        """Move to adjacent cell with split collision tracking."""
        r, c = target
        h, w = ground_truth.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        if ground_truth[r, c] == 100:
            self.collisions_static += 1
            return False
        for uid, opos in other_positions.items():
            if uid != self.uav_id and opos == (r, c):
                self.collisions_inter_uav += 1
        self._pos = (r, c)
        return True

    def receive_layer2_delta(self, delta: Layer2Delta) -> None:
        self.mask.merge_delta(delta)
