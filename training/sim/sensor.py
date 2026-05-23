"""
2D ray-casting sensor matching production ray_point from
uav_gogogo_continue_onnx.py:737-743 (vectorized np.linspace interpolation).

Returns visible cells within view_range using 360-degree ray cast.
"""
from __future__ import annotations

import numpy as np

from .config import SensorConfig


def ray_point(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """
    Vectorized ray tracing from (x0,y0) to (x1,y1).
    Matches production: Chebyshev steps + np.linspace interpolation.

    Returns (N, 2) array of integer grid coordinates along the ray.
    """
    num_steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    if num_steps <= 1:
        return np.array([[x0, y0]], dtype=np.int32)
    t = np.linspace(0, 1, num_steps)
    x = np.round(x0 * (1 - t) + x1 * t).astype(np.int32)
    y = np.round(y0 * (1 - t) + y1 * t).astype(np.int32)
    return np.column_stack((x, y))


def cast_rays(
    pos_x: int,
    pos_y: int,
    ground_truth: np.ndarray,
    cfg: SensorConfig | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Cast rays from UAV position and return (free_cells, occupied_cells).

    Uses ground truth to determine what the sensor "sees":
    - Free cells along each ray until hitting an obstacle or range limit
    - The first occupied cell hit (if any)

    Args:
        pos_x: UAV column position in global frame
        pos_y: UAV row position in global frame
        ground_truth: 2D array (height, width) with FREE=0, OCCUPIED=100
        cfg: Sensor configuration

    Returns:
        (free_cells, occupied_cells) as sets of (row, col) tuples
    """
    cfg = cfg or SensorConfig()
    h, w = ground_truth.shape
    free_cells: set[tuple[int, int]] = set()
    occupied_cells: set[tuple[int, int]] = set()

    angles = np.linspace(0, 2 * np.pi, cfg.num_rays, endpoint=False)
    end_x = pos_x + (cfg.view_range * np.cos(angles)).astype(int)
    end_y = pos_y + (cfg.view_range * np.sin(angles)).astype(int)

    end_x = np.clip(end_x, 0, w - 1)
    end_y = np.clip(end_y, 0, h - 1)

    for ex, ey in zip(end_x, end_y):
        points = ray_point(pos_x, pos_y, int(ex), int(ey))
        for px, py in points:
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            if ground_truth[py, px] == 100:
                occupied_cells.add((py, px))
                break
            free_cells.add((py, px))

    return free_cells, occupied_cells
