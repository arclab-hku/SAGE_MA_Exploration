"""
Communication channel with byte tracking, latency, and digest/snapshot (A2).

A7 correction: baseline and proposed each get independent channel instances
sharing the same pre-generated noise sequence (not the same object).
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import CommConfig


@dataclass
class ChannelStats:
    """Per-message-type byte counters."""
    total_bytes: int = 0
    message_counts: Dict[str, int] = field(default_factory=dict)
    byte_counts: Dict[str, int] = field(default_factory=dict)

    def record(self, msg_type: str, byte_size: int) -> None:
        self.total_bytes += byte_size
        self.message_counts[msg_type] = self.message_counts.get(msg_type, 0) + 1
        self.byte_counts[msg_type] = self.byte_counts.get(msg_type, 0) + byte_size


class CommChannel:
    """Simulated communication channel between UAVs.

    Features:
      - Configurable latency (messages arrive after N steps)
      - Optional packet loss with pre-generated noise (A7)
      - Byte tracking per message type
      - Periodic digest for divergence detection (A2)
    """

    def __init__(
        self,
        cfg: CommConfig,
        noise_rng: Optional[np.random.RandomState] = None,
    ):
        self._cfg = cfg
        self._rng = noise_rng or np.random.RandomState(0)
        # Pre-generate loss decisions for determinism (A7)
        self._loss_sequence = self._rng.random(10000)
        self._loss_idx = 0

        # Delayed message queue: list of (deliver_at_step, src, dst, msg_type, msg, byte_size)
        self._queue: deque = deque()
        self._stats_per_uav: Dict[int, ChannelStats] = {}

    def stats(self, uav_id: int) -> ChannelStats:
        if uav_id not in self._stats_per_uav:
            self._stats_per_uav[uav_id] = ChannelStats()
        return self._stats_per_uav[uav_id]

    @property
    def total_bytes(self) -> int:
        return sum(s.total_bytes for s in self._stats_per_uav.values())

    def send(
        self,
        current_step: int,
        src_uav: int,
        dst_uav: int,
        msg_type: str,
        msg: Any,
        byte_size: int,
    ) -> bool:
        """Enqueue a message. Returns False if dropped by packet loss."""
        # Check packet loss
        if self._cfg.loss_rate > 0:
            idx = self._loss_idx % len(self._loss_sequence)
            self._loss_idx += 1
            if self._loss_sequence[idx] < self._cfg.loss_rate:
                return False

        deliver_at = current_step + self._cfg.latency_steps
        self._queue.append((deliver_at, src_uav, dst_uav, msg_type, msg, byte_size))

        # Track stats for sender
        self.stats(src_uav).record(msg_type, byte_size)
        return True

    def broadcast(
        self,
        current_step: int,
        src_uav: int,
        all_uav_ids: List[int],
        msg_type: str,
        msg: Any,
        byte_size: int,
    ) -> int:
        """Broadcast to all other UAVs. Returns number successfully sent."""
        sent = 0
        for dst in all_uav_ids:
            if dst != src_uav:
                if self.send(current_step, src_uav, dst, msg_type, msg, byte_size):
                    sent += 1
        return sent

    def receive(self, current_step: int, dst_uav: int) -> List[Tuple[str, Any]]:
        """Collect all messages ready for delivery to dst_uav."""
        ready = []
        remaining = deque()
        while self._queue:
            item = self._queue.popleft()
            deliver_at, src, dst, msg_type, msg, byte_size = item
            if deliver_at <= current_step and dst == dst_uav:
                ready.append((msg_type, msg))
            else:
                remaining.append(item)
        self._queue = remaining
        return ready

    def pending_count(self) -> int:
        return len(self._queue)
