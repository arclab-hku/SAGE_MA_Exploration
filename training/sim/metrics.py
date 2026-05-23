"""
Extended metrics collection (req #5 + A7 fair comparison).

Tracks per-step:
  - Coverage ratio
  - Communication bytes (cumulative)
  - Collision count
  - Replan count
  - Frontier expiration count
  - Graph size (proposed only)
  - Stale frontier count (proposed only)
  - Task completion time (steps to 95% coverage)
  - Success rate (whether 95% coverage reached)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepMetrics:
    step: int
    coverage: float
    comm_bytes_cumulative: int
    collisions: int
    replans: int
    frontier_expirations: int
    graph_node_count: int = 0
    stale_frontiers: int = 0
    collisions_static: int = 0
    collisions_inter_uav: int = 0


@dataclass
class RunMetrics:
    """Full metrics for one run (baseline or proposed)."""
    system_name: str
    map_name: str
    num_uavs: int
    steps: List[StepMetrics] = field(default_factory=list)
    completion_step: Optional[int] = None  # step when 95% coverage reached
    final_coverage: float = 0.0
    success: bool = False  # whether 95% coverage was reached

    def record_step(self, sm: StepMetrics) -> None:
        self.steps.append(sm)
        if sm.coverage >= 0.95 and self.completion_step is None:
            self.completion_step = sm.step
            self.success = True
        self.final_coverage = sm.coverage

    @property
    def total_comm_bytes(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].comm_bytes_cumulative

    @property
    def total_collisions(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].collisions

    @property
    def collision_rate(self) -> float:
        if not self.steps:
            return 0.0
        total_steps = len(self.steps)
        return self.total_collisions / total_steps if total_steps > 0 else 0.0

    @property
    def total_replans(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].replans

    @property
    def total_frontier_expirations(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].frontier_expirations

    def coverage_at_step(self, step: int) -> float:
        if step < len(self.steps):
            return self.steps[step].coverage
        return self.final_coverage

    def comm_bytes_at_step(self, step: int) -> int:
        if step < len(self.steps):
            return self.steps[step].comm_bytes_cumulative
        return self.total_comm_bytes

    @property
    def total_collisions_static(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].collisions_static

    @property
    def total_collisions_inter_uav(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].collisions_inter_uav

    def summary(self) -> Dict:
        return {
            "system": self.system_name,
            "map": self.map_name,
            "num_uavs": self.num_uavs,
            "final_coverage": round(self.final_coverage, 4),
            "completion_step": self.completion_step,
            "success": self.success,
            "total_comm_bytes": self.total_comm_bytes,
            "comm_bytes_per_step": (
                self.total_comm_bytes / len(self.steps) if self.steps else 0
            ),
            "collision_rate": round(self.collision_rate, 4),
            "collisions_static": self.total_collisions_static,
            "collisions_inter_uav": self.total_collisions_inter_uav,
            "total_replans": self.total_replans,
            "total_frontier_expirations": self.total_frontier_expirations,
        }
