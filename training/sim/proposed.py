"""
Proposed system: Layer1(local) + Layer2(incremental delta) + Layer3(GraphDelta).

Communication via Layer2Delta + GraphDelta instead of ChunkData chunks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .comm import CommChannel
from .config import (
    CommConfig,
    GraphConfig,
    Layer2Config,
    PlannerConfig,
    SensorConfig,
    WorldConfig,
)
from .crdt import GraphDelta
from .explored_mask import Layer2Delta
from .uav import UAVAgent


class ProposedSystem:
    """Run the proposed 3-layer architecture with multiple UAVs."""

    def __init__(
        self,
        uav_positions: List[Tuple[int, int]],
        wcfg: WorldConfig,
        scfg: SensorConfig,
        l2cfg: Layer2Config,
        gcfg: GraphConfig,
        pcfg: PlannerConfig,
        ccfg: CommConfig,
        noise_rng: np.random.RandomState,
    ):
        self._wcfg = wcfg
        self._ccfg = ccfg
        self._uav_ids = list(range(len(uav_positions)))

        self.uavs: Dict[int, UAVAgent] = {}
        for uid, pos in enumerate(uav_positions):
            self.uavs[uid] = UAVAgent(
                uav_id=uid, start_pos=pos,
                wcfg=wcfg, scfg=scfg, l2cfg=l2cfg, gcfg=gcfg, pcfg=pcfg,
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
            other_pos = {k: v for k, v in positions.items() if k != uid}
            l2_delta, g_delta = uav.step(
                ground_truth, current_step, self._uav_ids, other_pos,
            )

            # Broadcast deltas
            if l2_delta is not None:
                self.channel.broadcast(
                    current_step, uid, self._uav_ids,
                    "layer2_delta", l2_delta, l2_delta.byte_size,
                )
            if g_delta is not None:
                self.channel.broadcast(
                    current_step, uid, self._uav_ids,
                    "graph_delta", g_delta, g_delta.byte_size,
                )

            # Periodic digest (A2)
            if (
                current_step > 0
                and current_step % self._ccfg.digest_interval == 0
            ):
                digest = uav.mask.compute_digest()
                self.channel.broadcast(
                    current_step, uid, self._uav_ids,
                    "digest", digest, digest.byte_size,
                )

        # Deliver messages
        for uid, uav in self.uavs.items():
            messages = self.channel.receive(current_step, uid)
            for msg_type, msg in messages:
                if msg_type == "layer2_delta":
                    uav.receive_layer2_delta(msg)
                elif msg_type == "graph_delta":
                    uav.receive_graph_delta(msg, current_step)
                # digest handled separately if needed

    def coverage_ratio(self) -> float:
        """Max coverage across all UAVs."""
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

    def graph_sizes(self) -> Dict[int, int]:
        return {uid: uav.graph.active_node_count() for uid, uav in self.uavs.items()}


class BaselineSystem:
    """Run the baseline (global shared grid) with multiple UAVs."""

    def __init__(
        self,
        uav_positions: List[Tuple[int, int]],
        wcfg: WorldConfig,
        scfg: SensorConfig,
        pcfg: PlannerConfig,
        ccfg: CommConfig,
        noise_rng: np.random.RandomState,
    ):
        from .baseline import BaselineUAV

        self._wcfg = wcfg
        self._ccfg = ccfg
        self._uav_ids = list(range(len(uav_positions)))

        self.uavs: Dict[int, BaselineUAV] = {}
        for uid, pos in enumerate(uav_positions):
            self.uavs[uid] = BaselineUAV(
                uav_id=uid, start_pos=pos,
                wcfg=wcfg, scfg=scfg, pcfg=pcfg,
            )

        # Each UAV has its own map copy — sync via chunks (fair comm comparison)

        self.channel = CommChannel(ccfg, noise_rng=noise_rng)

    def step(
        self,
        ground_truth: np.ndarray,
        current_step: int,
    ) -> None:
        for uid, uav in self.uavs.items():
            chunks = uav.step(ground_truth, current_step)
            for chunk in chunks:
                self.channel.broadcast(
                    current_step, uid, self._uav_ids,
                    "chunk", chunk, chunk.byte_size,
                )

        # Deliver chunks (in baseline, map is already shared,
        # but we track bandwidth for fair comparison)
        for uid, uav in self.uavs.items():
            messages = self.channel.receive(current_step, uid)
            for msg_type, msg in messages:
                if msg_type == "chunk":
                    uav.apply_chunk(msg)

    def coverage_ratio(self) -> float:
        return max(uav.coverage_ratio() for uav in self.uavs.values())

    def total_comm_bytes(self) -> int:
        # Channel bytes + idx_list stamp overhead (production protocol)
        stamp_overhead = sum(uav._stamp_bytes for uav in self.uavs.values())
        return self.channel.total_bytes + stamp_overhead

    def total_collisions(self) -> int:
        return sum(uav.collisions for uav in self.uavs.values())
