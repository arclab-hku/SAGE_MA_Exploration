"""Tests for sensor module: ray_point + cast_rays."""
from __future__ import annotations

import numpy as np
import pytest

from sim.config import SensorConfig, WorldConfig
from sim.sensor import cast_rays, ray_point
from sim.world import make_world


class TestRayPoint:
    """Vectorized ray tracing matches production."""

    def test_horizontal_ray(self):
        pts = ray_point(0, 0, 5, 0)
        assert pts.shape == (6, 2)
        np.testing.assert_array_equal(pts[:, 0], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(pts[:, 1], [0, 0, 0, 0, 0, 0])

    def test_vertical_ray(self):
        pts = ray_point(0, 0, 0, 5)
        assert pts.shape == (6, 2)
        np.testing.assert_array_equal(pts[:, 0], [0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(pts[:, 1], [0, 1, 2, 3, 4, 5])

    def test_diagonal_ray(self):
        pts = ray_point(0, 0, 4, 4)
        assert pts.shape == (5, 2)
        np.testing.assert_array_equal(pts[:, 0], pts[:, 1])

    def test_single_point(self):
        pts = ray_point(3, 3, 3, 3)
        assert pts.shape == (1, 2)
        np.testing.assert_array_equal(pts[0], [3, 3])

    def test_negative_direction(self):
        pts = ray_point(5, 5, 0, 0)
        assert pts[0, 0] == 5
        assert pts[-1, 0] == 0
        assert len(pts) == 6

    def test_returns_int_coordinates(self):
        pts = ray_point(0, 0, 7, 3)
        assert pts.dtype == np.int32


class TestCastRays:
    """360-degree ray casting in ground truth grid."""

    def test_open_area_sees_all_nearby(self):
        grid = np.zeros((20, 20), dtype=np.int8)
        grid[0, :] = 100
        grid[-1, :] = 100
        grid[:, 0] = 100
        grid[:, -1] = 100

        cfg = SensorConfig(view_range=5, num_rays=72)
        free, occ = cast_rays(10, 10, grid, cfg)
        assert (10, 10) in free  # UAV position is free
        assert len(free) > 0

    def test_wall_blocks_rays(self):
        grid = np.zeros((20, 20), dtype=np.int8)
        # Wall at column 12
        grid[:, 12] = 100

        cfg = SensorConfig(view_range=10, num_rays=72)
        free, occ = cast_rays(10, 10, grid, cfg)
        # Should see the wall
        assert any(c == 12 for _, c in occ)
        # Should NOT see behind the wall
        assert all(c <= 12 for _, c in free | occ)

    def test_corner_position(self):
        grid = np.zeros((20, 20), dtype=np.int8)
        grid[0, :] = 100
        grid[:, 0] = 100

        cfg = SensorConfig(view_range=5, num_rays=72)
        free, occ = cast_rays(1, 1, grid, cfg)
        assert len(free) > 0
        assert len(occ) > 0  # should see border walls

    def test_sensor_range_limit(self):
        grid = np.zeros((80, 80), dtype=np.int8)
        cfg = SensorConfig(view_range=10, num_rays=72)
        free, occ = cast_rays(40, 40, grid, cfg)
        # All visible cells should be within view_range
        for r, c in free:
            dist = max(abs(c - 40), abs(r - 40))
            assert dist <= cfg.view_range + 1  # +1 for rounding

    def test_no_duplicate_cells(self):
        grid = np.zeros((20, 20), dtype=np.int8)
        cfg = SensorConfig(view_range=5, num_rays=72)
        free, occ = cast_rays(10, 10, grid, cfg)
        # free and occ should not overlap
        assert len(free & occ) == 0

    def test_on_preset_map(self):
        grid = make_world("rooms")
        cfg = SensorConfig(view_range=10, num_rays=72)
        free, occ = cast_rays(10, 10, grid, cfg)
        assert len(free) > 0
