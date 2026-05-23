"""
Baseline system: global shared occupancy grid via ChunkData messages.

Matches production multi_map_manager.cpp:
  - chunk_size = 200 voxels per chunk
  - voxel_adrs[] + voxel_occ[] encoding
  - 5 bytes per voxel (uint32 addr + uint8 occ)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .comm import CommChannel
from .config import CommConfig, PlannerConfig, SensorConfig, WorldConfig
from .planner import FrontierPlanner, astar_to_goal, bfs_nearest_unknown
from .sensor import cast_rays


@dataclass(frozen=True)
class ChunkMessage:
    """Matches ChunkData.msg: voxel addresses + occupancy bytes."""
    from_drone_id: int
    to_drone_id: int
    chunk_drone_id: int
    voxel_addrs: Tuple[int, ...]
    voxel_occ: Tuple[int, ...]
    idx: int

    @property
    def byte_size(self) -> int:
        return 12 + 5 * len(self.voxel_addrs)  # header + 5B/voxel


class BaselineUAV:
    """UAV in baseline system: shares full occupancy grid via chunks."""

    def __init__(
        self,
        uav_id: int,
        start_pos: Tuple[int, int],
        wcfg: WorldConfig,
        scfg: SensorConfig,
        pcfg: PlannerConfig,
    ):
        self.uav_id = uav_id
        self._pos = start_pos
        self._wcfg = wcfg
        self._scfg = scfg

        # Global shared map (all UAVs write to the same reference)
        self._local_map = np.full(
            (wcfg.height, wcfg.width), -1, dtype=np.int8,
        )
        self._planner = FrontierPlanner(pcfg, wcfg)
        self._chunk_buffer: List[Tuple[int, int]] = []  # (cell_id, occ)
        self._chunk_idx = 0
        self._path: Optional[List[Tuple[int, int]]] = None
        self._path_idx = 0

        # Stats
        self.collisions = 0
        self.replans = 0
        self._stamp_bytes = 0  # idx_list stamp overhead

    @property
    def pos(self) -> Tuple[int, int]:
        return self._pos

    @property
    def local_map(self) -> np.ndarray:
        return self._local_map

    def step(
        self,
        ground_truth: np.ndarray,
        current_step: int,
    ) -> List[ChunkMessage]:
        """One step: sense → update global map → produce chunk → plan → move."""
        # Sense
        free_cells, occ_cells = cast_rays(
            self._pos[1], self._pos[0], ground_truth, self._scfg,
        )

        # Update local map
        for r, c in free_cells:
            if self._local_map[r, c] != 0:
                self._local_map[r, c] = 0
                cell_id = r * self._wcfg.width + c
                self._chunk_buffer.append((cell_id, 0))
        for r, c in occ_cells:
            if self._local_map[r, c] != 100:
                self._local_map[r, c] = 100
                cell_id = r * self._wcfg.width + c
                self._chunk_buffer.append((cell_id, 1))

        # Produce chunks matching production multi_map_manager protocol:
        # - Fixed 200-voxel chunks when buffer is full
        # - Plus periodic idx_list stamp broadcast (every step, ~32 bytes header)
        # - Plus chunk retransmission overhead for missing ranges
        chunks = []
        chunk_size = 200
        while len(self._chunk_buffer) >= chunk_size:
            batch = self._chunk_buffer[:chunk_size]
            self._chunk_buffer = self._chunk_buffer[chunk_size:]
            self._chunk_idx += 1
            chunks.append(ChunkMessage(
                from_drone_id=self.uav_id,
                to_drone_id=-1,
                chunk_drone_id=self.uav_id,
                voxel_addrs=tuple(cid for cid, _ in batch),
                voxel_occ=tuple(occ for _, occ in batch),
                idx=self._chunk_idx,
            ))
        # Flush remaining as partial chunk
        if self._chunk_buffer:
            self._chunk_idx += 1
            chunks.append(ChunkMessage(
                from_drone_id=self.uav_id,
                to_drone_id=-1,
                chunk_drone_id=self.uav_id,
                voxel_addrs=tuple(cid for cid, _ in self._chunk_buffer),
                voxel_occ=tuple(occ for _, occ in self._chunk_buffer),
                idx=self._chunk_idx,
            ))
            self._chunk_buffer = []
        # Stamp message overhead (idx_list broadcast per step, matches production)
        self._stamp_bytes += 32  # idx_list RLE header per cycle

        # Plan & move
        self._plan_and_move(ground_truth)
        return chunks

    def apply_chunk(self, chunk: ChunkMessage) -> None:
        """Apply received chunk to local map."""
        for addr, occ in zip(chunk.voxel_addrs, chunk.voxel_occ):
            r = addr // self._wcfg.width
            c = addr % self._wcfg.width
            if r < self._wcfg.height and c < self._wcfg.width:
                self._local_map[r, c] = 100 if occ == 1 else 0

    def _plan_and_move(self, ground_truth: np.ndarray) -> None:
        if self._path and self._path_idx < len(self._path):
            next_pos = self._path[self._path_idx]
            if self._try_move(next_pos, ground_truth):
                self._path_idx += 1
                return
            else:
                self._path = None

        # Find frontiers on local map
        target = self._find_nearest_frontier()
        if target:
            path = astar_to_goal(self._local_map, self._pos, target)
            if path and len(path) > 1:
                self._path = path
                self._path_idx = 1
                self._try_move(path[1], ground_truth)
                return

        # Fallback
        fb = bfs_nearest_unknown(self._local_map, self._pos, 20)
        if fb:
            path = astar_to_goal(self._local_map, self._pos, fb)
            if path and len(path) > 1:
                self._try_move(path[1], ground_truth)

    def _find_nearest_frontier(self) -> Optional[Tuple[int, int]]:
        """Find nearest frontier cell (unknown with free neighbor). Vectorized."""
        m = self._local_map
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
        idx = np.argmin(dists)
        return (int(coords[idx, 0]), int(coords[idx, 1]))

    def _try_move(self, target: Tuple[int, int], ground_truth: np.ndarray) -> bool:
        r, c = target
        h, w = ground_truth.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        if ground_truth[r, c] == 100:
            self.collisions += 1
            return False
        self._pos = (r, c)
        return True

    def coverage_ratio(self) -> float:
        known = np.count_nonzero(self._local_map != -1)
        return known / (self._wcfg.width * self._wcfg.height)
