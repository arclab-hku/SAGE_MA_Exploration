"""
3-state explored mask with confidence tracking (Layer 2).

Adjustments applied:
  A1: Version = (lamport_counter, source_uav) for deterministic tie-breaking
  A3: Merge uses obs_count — low-confidence occupied cannot override high-confidence free

States: UNKNOWN=-1, FREE=0, OCCUPIED=100
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import Layer2Config, WorldConfig


# ---------------------------------------------------------------------------
# Version tuple: (lamport_counter, source_uav) — A1
# ---------------------------------------------------------------------------

VersionT = Tuple[int, int]  # (lamport, source_uav)

_ZERO_VERSION: VersionT = (0, -1)


def _version_gt(a: VersionT, b: VersionT) -> bool:
    """a > b: compare lamport first, then source_uav for deterministic tie-break."""
    return a > b  # tuple comparison: (lamport, src) is naturally correct


# ---------------------------------------------------------------------------
# Delta messages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellUpdate:
    """Single cell in a Layer2Delta. Uses linear cell_id (no uint8 overflow)."""
    cell_id: int
    state: int
    obs_count: int
    version: VersionT


@dataclass(frozen=True)
class Layer2Delta:
    """Incremental sync message for the global 2D grid.

    Byte size computed via RLE compression:
    - Sorted by cell_id, consecutive cells with same state form a run
    - Run header: start_cell_id(2) + count(1) + state(1) + obs_count(1) + version(2) = 7 bytes
    - vs naive: 6 bytes per cell
    """
    source_uav: int
    seq: int
    base_version: int
    changed_cells: tuple  # tuple[CellUpdate, ...]

    @property
    def byte_size(self) -> int:
        if not self.changed_cells:
            return 8  # header only
        return 8 + self._rle_byte_size()

    def _rle_byte_size(self) -> int:
        """Compute byte size using RLE encoding of sorted cells.
        Each run: start_cell_id(2) + count(1) + state(1) + obs_count(1) + version(2) = 7 bytes."""
        return self._count_rle_runs() * 7

    def _count_rle_runs(self) -> int:
        """Count RLE runs: groups of consecutive cell_ids with same state."""
        if not self.changed_cells:
            return 0
        cells = sorted(self.changed_cells, key=lambda c: c.cell_id)
        runs = 1
        for j in range(1, len(cells)):
            prev = cells[j - 1]
            curr = cells[j]
            if curr.cell_id != prev.cell_id + 1 or curr.state != prev.state:
                runs += 1
        return runs


@dataclass(frozen=True)
class StateDigest:
    """Periodic hash of full grid state for divergence detection (A2)."""
    source_uav: int
    seq: int
    digest: bytes  # 16-byte MD5

    @property
    def byte_size(self) -> int:
        return 32


@dataclass(frozen=True)
class SnapshotRequest:
    """Request full state for a region after digest mismatch (A2)."""
    requester_uav: int
    cell_ids: tuple[int, ...]


@dataclass(frozen=True)
class SnapshotResponse:
    """Full state for requested cells (A2)."""
    source_uav: int
    cells: tuple[CellUpdate, ...]

    @property
    def byte_size(self) -> int:
        return 8 + 6 * len(self.cells)


# ---------------------------------------------------------------------------
# Explored Mask
# ---------------------------------------------------------------------------

class ExploredMask:
    """Per-UAV Layer2 mask with confidence and CRDT merge."""

    def __init__(
        self,
        cfg: WorldConfig,
        source_uav: int,
        l2_cfg: Layer2Config | None = None,
    ):
        self._cfg = cfg
        self._source = source_uav
        self._l2 = l2_cfg or Layer2Config()
        self._width = cfg.width
        self._height = cfg.height

        self._state = np.full(
            (cfg.height, cfg.width), self._l2.UNKNOWN, dtype=np.int8,
        )
        self._obs_count = np.zeros((cfg.height, cfg.width), dtype=np.int8)
        # Per-cell version: stored as separate lamport + source arrays
        self._ver_lamport = np.zeros((cfg.height, cfg.width), dtype=np.uint32)
        self._ver_source = np.full(
            (cfg.height, cfg.width), -1, dtype=np.int16,
        )
        self._lamport_clock = 0  # local Lamport counter
        self._seq = 0
        self._dirty: set[tuple[int, int]] = set()
        self._defer_count = 0  # steps since last flush

    # -- properties --

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def obs_count(self) -> np.ndarray:
        return self._obs_count

    def cell_version(self, row: int, col: int) -> VersionT:
        return (int(self._ver_lamport[row, col]), int(self._ver_source[row, col]))

    def coverage_ratio(self) -> float:
        known = np.count_nonzero(self._state != self._l2.UNKNOWN)
        return known / (self._width * self._height)

    # -- local sensor update --

    def update_from_sensor(
        self,
        free_cells: set[tuple[int, int]],
        occupied_cells: set[tuple[int, int]],
    ) -> None:
        for row, col in free_cells:
            self._update_cell(row, col, self._l2.FREE)
        for row, col in occupied_cells:
            self._update_cell(row, col, self._l2.OCCUPIED)

    def _update_cell(self, row: int, col: int, new_state: int) -> None:
        old_state = self._state[row, col]
        old_count = int(self._obs_count[row, col])

        if old_state == new_state:
            # Reinforce confidence locally — no version bump, no delta
            new_count = min(old_count + 1, self._l2.max_observation_count)
            if new_count != old_count:
                self._obs_count[row, col] = new_count
                # No _bump_version: reinforcement must NOT inflate Lamport clock
        else:
            # State transition — needs sync
            self._state[row, col] = new_state
            self._obs_count[row, col] = 1
            self._bump_version(row, col)
            self._dirty.add((row, col))

    def _bump_version(self, row: int, col: int) -> None:
        self._lamport_clock += 1
        self._ver_lamport[row, col] = self._lamport_clock
        self._ver_source[row, col] = self._source

    # -- decay (A3: prevents sticky false obstacles) --

    def decay_tick(self) -> None:
        occ_mask = self._state == self._l2.OCCUPIED
        if not np.any(occ_mask):
            return

        rows, cols = np.where(occ_mask)
        for r, c in zip(rows, cols):
            new_count = max(0, int(self._obs_count[r, c]) - self._l2.decay_amount)
            if new_count <= self._l2.clear_threshold:
                self._state[r, c] = self._l2.UNKNOWN
                self._obs_count[r, c] = 0
                self._bump_version(r, c)
                self._dirty.add((r, c))
            elif new_count != self._obs_count[r, c]:
                self._obs_count[r, c] = new_count
                self._dirty.add((r, c))

    # -- delta production --

    def flush_delta(self) -> Layer2Delta | None:
        if not self._dirty:
            self._defer_count = 0
            return None

        self._defer_count += 1

        # Minimum delta threshold: defer small deltas unless max defer reached
        if (
            len(self._dirty) < self._l2.min_delta_cells
            and self._defer_count < self._l2.max_delta_defer_steps
        ):
            return None

        cells = tuple(
            CellUpdate(
                cell_id=row * self._width + col,
                state=int(self._state[row, col]),
                obs_count=int(self._obs_count[row, col]),
                version=(
                    int(self._ver_lamport[row, col]),
                    int(self._ver_source[row, col]),
                ),
            )
            for row, col in self._dirty
        )
        self._seq += 1
        delta = Layer2Delta(
            source_uav=self._source,
            seq=self._seq,
            base_version=self._seq - 1,
            changed_cells=cells,
        )
        self._dirty.clear()
        self._defer_count = 0
        return delta

    # -- CRDT merge (A1 version tuple + A3 obs_count aware) --

    def merge_delta(self, delta: Layer2Delta) -> None:
        # Advance local Lamport clock
        for cell in delta.changed_cells:
            if cell.version[0] > self._lamport_clock:
                self._lamport_clock = cell.version[0]

        for cell in delta.changed_cells:
            col = cell.cell_id % self._width
            row = cell.cell_id // self._width
            if row >= self._height or col >= self._width:
                continue

            local_ver = self.cell_version(row, col)
            remote_ver = cell.version

            if _version_gt(remote_ver, local_ver):
                # Safety guard: don't let remote FREE override local OCCUPIED
                # unless remote has sufficient observations (A3)
                local_state = int(self._state[row, col])
                if (
                    local_state == self._l2.OCCUPIED
                    and cell.state != self._l2.OCCUPIED
                    and cell.obs_count < self._l2.merge_min_obs_to_override
                ):
                    pass  # keep local OCCUPIED — remote lacks confidence
                else:
                    self._accept_remote(row, col, cell)
            elif remote_ver == local_ver:
                # Same version tie-break: state priority + obs_count (A3)
                self._merge_same_version(row, col, cell)
            # else: local is newer, ignore

    def _accept_remote(self, row: int, col: int, cell: CellUpdate) -> None:
        self._state[row, col] = cell.state
        self._obs_count[row, col] = cell.obs_count
        self._ver_lamport[row, col] = cell.version[0]
        self._ver_source[row, col] = cell.version[1]

    def _merge_same_version(self, row: int, col: int, cell: CellUpdate) -> None:
        """Same version tie-break (A3):
        - State priority: occupied > free > unknown
        - But low-confidence occupied (obs_count < threshold) cannot override
          high-confidence free."""
        remote_pri = _state_priority(cell.state)
        local_pri = _state_priority(int(self._state[row, col]))

        if remote_pri > local_pri:
            # A3: remote must have enough observations to override
            if cell.obs_count >= self._l2.merge_min_obs_to_override:
                self._accept_remote(row, col, cell)
            # else: low-confidence remote, keep local
        elif remote_pri == local_pri:
            # Same state: take higher obs_count
            if cell.obs_count > self._obs_count[row, col]:
                self._accept_remote(row, col, cell)

    # -- digest for divergence detection (A2) --

    def compute_digest(self) -> StateDigest:
        h = hashlib.md5()
        h.update(self._state.tobytes())
        h.update(self._obs_count.tobytes())
        return StateDigest(
            source_uav=self._source,
            seq=self._seq,
            digest=h.digest(),
        )

    def build_snapshot_response(self, cell_ids: tuple[int, ...]) -> SnapshotResponse:
        cells = []
        for cid in cell_ids:
            col = cid % self._width
            row = cid // self._width
            if row < self._height and col < self._width:
                cells.append(CellUpdate(
                    cell_id=cid,
                    state=int(self._state[row, col]),
                    obs_count=int(self._obs_count[row, col]),
                    version=(
                        int(self._ver_lamport[row, col]),
                        int(self._ver_source[row, col]),
                    ),
                ))
        return SnapshotResponse(source_uav=self._source, cells=tuple(cells))

    def apply_snapshot(self, snap: SnapshotResponse) -> None:
        """Apply snapshot cells using same CRDT merge rules."""
        for cell in snap.cells:
            col = cell.cell_id % self._width
            row = cell.cell_id // self._width
            if row >= self._height or col >= self._width:
                continue
            local_ver = self.cell_version(row, col)
            if _version_gt(cell.version, local_ver):
                self._accept_remote(row, col, cell)


def _state_priority(state: int) -> int:
    if state == 100:
        return 2
    if state == 0:
        return 1
    return 0
