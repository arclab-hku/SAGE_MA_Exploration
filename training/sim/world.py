"""
Ground-truth 2D world grid with preset map generators.

All coordinates are in a unified global frame (requirement #1).
Origin = (0, 0) at top-left corner; no per-UAV transforms needed.
"""
from __future__ import annotations

import numpy as np

from .config import WorldConfig

FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def make_world(map_name: str, cfg: WorldConfig | None = None) -> np.ndarray:
    """Return a ground-truth grid for the given map preset."""
    cfg = cfg or WorldConfig()
    builders = {
        "rooms": _build_rooms,
        "maze": _build_maze,
        "open": _build_open,
    }
    if map_name not in builders:
        raise ValueError(f"Unknown map: {map_name}. Choose from {list(builders)}")
    return builders[map_name](cfg)


def _border(grid: np.ndarray) -> np.ndarray:
    """Set border cells to occupied."""
    grid[0, :] = OCCUPIED
    grid[-1, :] = OCCUPIED
    grid[:, 0] = OCCUPIED
    grid[:, -1] = OCCUPIED
    return grid


def _build_rooms(cfg: WorldConfig) -> np.ndarray:
    """4 rooms connected by narrow corridors."""
    grid = np.full((cfg.height, cfg.width), FREE, dtype=np.int8)
    grid = _border(grid)

    h, w = cfg.height, cfg.width
    mid_h, mid_w = h // 2, w // 2

    # Horizontal wall
    grid[mid_h, 1:w - 1] = OCCUPIED
    # Vertical wall
    grid[1:h - 1, mid_w] = OCCUPIED

    # Corridors (3-cell wide gaps)
    corridor_w = 3
    # Top-bottom corridor in left half
    c1 = mid_w // 2
    grid[mid_h, c1 - 1:c1 + corridor_w - 1] = FREE
    # Top-bottom corridor in right half
    c2 = mid_w + mid_w // 2
    grid[mid_h, c2 - 1:c2 + corridor_w - 1] = FREE
    # Left-right corridor in top half
    r1 = mid_h // 2
    grid[r1 - 1:r1 + corridor_w - 1, mid_w] = FREE
    # Left-right corridor in bottom half
    r2 = mid_h + mid_h // 2
    grid[r2 - 1:r2 + corridor_w - 1, mid_w] = FREE

    return grid


def _build_maze(cfg: WorldConfig) -> np.ndarray:
    """Maze-like corridors generated via recursive division."""
    grid = np.full((cfg.height, cfg.width), FREE, dtype=np.int8)
    grid = _border(grid)

    rng = np.random.RandomState(12345)
    _recursive_division(grid, 1, 1, cfg.width - 2, cfg.height - 2, rng, depth=0)

    return grid


def _recursive_division(
    grid: np.ndarray,
    x: int, y: int, w: int, h: int,
    rng: np.random.RandomState,
    depth: int,
) -> None:
    """Recursively divide space with walls + gaps to form a maze."""
    min_room = 6
    if w < min_room or h < min_room or depth > 6:
        return

    half = min_room // 2
    if w >= h:
        lo, hi = half, w - half
        if lo >= hi:
            return
        wall_x = x + rng.randint(lo, hi)
        gap_y = y + rng.randint(0, h)
        for row in range(y, y + h):
            grid[row, wall_x] = OCCUPIED
        for dy in range(-1, 2):
            gy = gap_y + dy
            if 0 < gy < grid.shape[0] - 1:
                grid[gy, wall_x] = FREE
        _recursive_division(grid, x, y, wall_x - x, h, rng, depth + 1)
        _recursive_division(grid, wall_x + 1, y, x + w - wall_x - 1, h, rng, depth + 1)
    else:
        lo, hi = half, h - half
        if lo >= hi:
            return
        wall_y = y + rng.randint(lo, hi)
        gap_x = x + rng.randint(0, w)
        for col in range(x, x + w):
            grid[wall_y, col] = OCCUPIED
        for dx in range(-1, 2):
            gx = gap_x + dx
            if 0 < gx < grid.shape[1] - 1:
                grid[wall_y, gx] = FREE
        _recursive_division(grid, x, y, w, wall_y - y, rng, depth + 1)
        _recursive_division(grid, x, wall_y + 1, w, y + h - wall_y - 1, rng, depth + 1)


def _build_open(cfg: WorldConfig) -> np.ndarray:
    """Open area with scattered rectangular obstacles."""
    grid = np.full((cfg.height, cfg.width), FREE, dtype=np.int8)
    grid = _border(grid)

    rng = np.random.RandomState(54321)
    num_obstacles = 12
    for _ in range(num_obstacles):
        ox = rng.randint(5, cfg.width - 10)
        oy = rng.randint(5, cfg.height - 10)
        ow = rng.randint(2, 6)
        oh = rng.randint(2, 6)
        grid[oy:oy + oh, ox:ox + ow] = OCCUPIED

    return grid


def cell_id_from_xy(x: int, y: int, width: int) -> int:
    """Convert (x, y) to linear cell_id. Requirement #2: supports large maps."""
    return y * width + x


def xy_from_cell_id(cell_id: int, width: int) -> tuple[int, int]:
    """Convert linear cell_id back to (x, y)."""
    return cell_id % width, cell_id // width
