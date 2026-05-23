"""
GBPlanner multi-UAV system with COHORT-style Voronoi partition.

Communication: Layer2Delta only (no GraphDelta — each UAV has its own RRG).
Uses CommChannel for fair bandwidth comparison with baseline and proposed.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .comm import CommChannel
from .config import (
    CommConfig,
    GBPlannerConfig,
    Layer2Config,
    PlannerConfig,
    SensorConfig,
    WorldConfig,
)
from .gbplanner_uav import GBPlannerUAV


class GBPlannerSystem:
    """Multi-UAV system using GBPlanner with COHORT-style Voronoi partition."""

    def __init__(
        self,
        uav_positions: List[Tuple[int, int]],
        wcfg: WorldConfig,
        scfg: SensorConfig,
        l2cfg: Layer2Config,
        pcfg: PlannerConfig,
        gbcfg: GBPlannerConfig,
        ccfg: CommConfig,
        noise_rng: np.random.RandomState,
    ):
        self._wcfg = wcfg
        self._ccfg = ccfg
        self._uav_ids = list(range(len(uav_positions)))

        self.uavs: Dict[int, GBPlannerUAV] = {}
        for uid, pos in enumerate(uav_positions):
            self.uavs[uid] = GBPlannerUAV(
                uav_id=uid, start_pos=pos,
                wcfg=wcfg, scfg=scfg, l2cfg=l2cfg, pcfg=pcfg, gbcfg=gbcfg,
            )

        self.channel = CommChannel(ccfg, noise_rng=noise_rng)

    def step(
        self,
        ground_truth: np.ndarray,
        current_step: int,
    ) -> None:
        """Run one step for all UAVs: sense, plan, communicate."""
        positions = {uid: uav.pos for uid, uav in self.uavs.items()}

        for uid, uav in self.uavs.items():
            l2_delta = uav.step(ground_truth, current_step, positions)

            # Broadcast Layer2Delta (no GraphDelta — each UAV has own RRG)
            if l2_delta is not None:
                self.channel.broadcast(
                    current_step, uid, self._uav_ids,
                    "layer2_delta", l2_delta, l2_delta.byte_size,
                )

        # Deliver messages
        for uid, uav in self.uavs.items():
            messages = self.channel.receive(current_step, uid)
            for msg_type, msg in messages:
                if msg_type == "layer2_delta":
                    uav.receive_layer2_delta(msg)

    def coverage_ratio(self) -> float:
        return max(uav.mask.coverage_ratio() for uav in self.uavs.values())

    def total_comm_bytes(self) -> int:
        return self.channel.total_bytes

    def total_collisions(self) -> int:
        return sum(uav.collisions for uav in self.uavs.values())

    def total_collisions_static(self) -> int:
        return sum(uav.collisions_static for uav in self.uavs.values())

    def total_collisions_inter_uav(self) -> int:
        return sum(uav.collisions_inter_uav for uav in self.uavs.values())

    def total_replans(self) -> int:
        return sum(uav.replans for uav in self.uavs.values())

    def total_frontier_expirations(self) -> int:
        return sum(uav.frontier_expirations for uav in self.uavs.values())
