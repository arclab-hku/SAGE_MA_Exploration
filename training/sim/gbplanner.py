"""
GBPlanner core: RRG graph construction, information-gain scoring, and Voronoi partition.

Adapted from the GBPlanner paper for 2D grid simulation:
  - RRG built via uniform random sampling in free cells
  - Information gain via 2D raycast counting unknown cells
  - Path cost via Dijkstra on RRG graph
  - Multi-robot coordination via Voronoi partition (nearest-robot assignment)
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GBPlannerConfig, SensorConfig
from .sensor import ray_point


@dataclass(frozen=True)
class RRGNode:
    """Node in the Rapidly-exploring Random Graph."""
    node_id: int
    x: int          # col
    y: int          # row
    gain: float     # information gain (unknown cells visible via raycast)


class RRGGraph:
    """Rapidly-exploring Random Graph for exploration planning."""

    def __init__(self, num_vertices_max: int) -> None:
        self.nodes: Dict[int, RRGNode] = {}
        self.adj: Dict[int, List[int]] = {}
        self.num_vertices_max = num_vertices_max
        self._next_id = 0

    def add_node(self, x: int, y: int, gain: float = 0.0) -> RRGNode:
        nid = self._next_id
        self._next_id += 1
        node = RRGNode(node_id=nid, x=x, y=y, gain=gain)
        self.nodes[nid] = node
        self.adj[nid] = []
        return node

    def add_edge(self, a: int, b: int) -> None:
        if b not in self.adj[a]:
            self.adj[a].append(b)
        if a not in self.adj[b]:
            self.adj[b].append(a)


def _collision_free(
    x0: int, y0: int, x1: int, y1: int, mask_state: np.ndarray,
) -> bool:
    """Check if the straight-line path between two cells is collision-free."""
    points = ray_point(x0, y0, x1, y1)
    h, w = mask_state.shape
    for px, py in points:
        if px < 0 or px >= w or py < 0 or py >= h:
            return False
        if mask_state[py, px] == 100:
            return False
    return True


def build_rrg(
    mask_state: np.ndarray,
    robot_pos: Tuple[int, int],
    rng: np.random.RandomState,
    cfg: GBPlannerConfig,
    scfg: SensorConfig,
    voronoi_mask: Optional[np.ndarray] = None,
) -> RRGGraph:
    """Build an RRG by sampling free cells in the Voronoi partition.

    Args:
        mask_state: (H, W) grid with -1=unknown, 0=free, 100=occupied.
        robot_pos: (row, col) of this robot.
        rng: seeded RNG for reproducibility.
        cfg: GBPlanner configuration.
        scfg: sensor config for gain computation.
        voronoi_mask: (H, W) bool mask of this robot's Voronoi region (optional).

    Returns:
        RRGGraph with nodes and edges.
    """
    graph = RRGGraph(cfg.num_vertices_max)

    # Add robot position as first node
    r, c = robot_pos
    robot_gain = _compute_gain_at(c, r, mask_state, scfg)
    robot_node = graph.add_node(x=c, y=r, gain=robot_gain)

    # Candidate free cells for sampling
    free_mask = mask_state == 0
    if voronoi_mask is not None:
        free_mask = free_mask & voronoi_mask
    free_coords = np.argwhere(free_mask)  # (N, 2) with [row, col]

    if len(free_coords) == 0:
        return graph

    # Sample up to num_samples_per_step candidates
    num_samples = min(cfg.num_samples_per_step, len(free_coords))
    indices = rng.choice(len(free_coords), size=num_samples, replace=False)
    samples = free_coords[indices]

    # Add sampled nodes (up to num_vertices_max - 1, since robot is already in)
    max_new = cfg.num_vertices_max - 1
    for i in range(min(len(samples), max_new)):
        sy, sx = int(samples[i, 0]), int(samples[i, 1])
        gain = _compute_gain_at(sx, sy, mask_state, scfg)
        graph.add_node(x=sx, y=sy, gain=gain)

    # Connect nodes: k-nearest + collision-free check
    _connect_rrg(graph, mask_state, cfg.k_nearest, cfg.max_edge_length)

    return graph


def _connect_rrg(
    graph: RRGGraph,
    mask_state: np.ndarray,
    k: int,
    max_edge_length: int,
) -> None:
    """Connect each node to its k-nearest collision-free neighbors."""
    node_ids = list(graph.nodes.keys())
    if len(node_ids) < 2:
        return

    coords = np.array(
        [(graph.nodes[nid].x, graph.nodes[nid].y) for nid in node_ids],
        dtype=np.float64,
    )

    for i, nid in enumerate(node_ids):
        dists = np.abs(coords[:, 0] - coords[i, 0]) + np.abs(coords[:, 1] - coords[i, 1])
        dists[i] = np.inf  # exclude self

        # Filter by max edge length
        valid = dists <= max_edge_length
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            continue

        # Sort valid by distance, take top-k
        sorted_valid = valid_indices[np.argsort(dists[valid_indices])]
        neighbors = sorted_valid[:k]

        for j in neighbors:
            other_id = node_ids[j]
            n1 = graph.nodes[nid]
            n2 = graph.nodes[other_id]
            if _collision_free(n1.x, n1.y, n2.x, n2.y, mask_state):
                graph.add_edge(nid, other_id)


def _compute_gain_at(
    x: int, y: int, mask_state: np.ndarray, scfg: SensorConfig,
) -> float:
    """Compute information gain at position (x=col, y=row) via 360-degree raycast."""
    h, w = mask_state.shape
    unknown_count = 0

    angles = np.linspace(0, 2 * np.pi, scfg.num_rays, endpoint=False)
    end_x = x + (scfg.view_range * np.cos(angles)).astype(int)
    end_y = y + (scfg.view_range * np.sin(angles)).astype(int)
    end_x = np.clip(end_x, 0, w - 1)
    end_y = np.clip(end_y, 0, h - 1)

    seen: set = set()
    for ex, ey in zip(end_x, end_y):
        points = ray_point(x, y, int(ex), int(ey))
        for px, py in points:
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            if mask_state[py, px] == 100:
                break
            cell = (py, px)
            if cell not in seen:
                seen.add(cell)
                if mask_state[py, px] == -1:
                    unknown_count += 1

    return float(unknown_count)


def compute_path_cost(
    graph: RRGGraph, start_id: int, target_id: int,
) -> float:
    """Dijkstra on RRG graph. Returns path cost (Manhattan edge weights) or inf."""
    if start_id not in graph.nodes or target_id not in graph.nodes:
        return float("inf")
    if start_id == target_id:
        return 0.0

    dist: Dict[int, float] = {start_id: 0.0}
    heap: list = [(0.0, start_id)]

    while heap:
        d, nid = heapq.heappop(heap)
        if nid == target_id:
            return d
        if d > dist.get(nid, float("inf")):
            continue
        n = graph.nodes[nid]
        for neighbor_id in graph.adj.get(nid, []):
            nb = graph.nodes[neighbor_id]
            edge_cost = abs(n.x - nb.x) + abs(n.y - nb.y)
            new_dist = d + edge_cost
            if new_dist < dist.get(neighbor_id, float("inf")):
                dist[neighbor_id] = new_dist
                heapq.heappush(heap, (new_dist, neighbor_id))

    return float("inf")


def select_best_vertex(
    graph: RRGGraph,
    robot_node_id: int,
    min_gain: float = 1.0,
) -> Optional[RRGNode]:
    """Greedy selection: max gain/cost among nodes with gain > threshold."""
    epsilon = 1e-6
    best_score = -1.0
    best_node: Optional[RRGNode] = None

    for nid, node in graph.nodes.items():
        if nid == robot_node_id:
            continue
        if node.gain < min_gain:
            continue
        cost = compute_path_cost(graph, robot_node_id, nid)
        if cost == float("inf"):
            continue
        score = node.gain / max(cost, epsilon)
        if score > best_score:
            best_score = score
            best_node = node

    return best_node


def voronoi_partition(
    robot_positions: Dict[int, Tuple[int, int]],
    mask_state: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute Voronoi partition: each free/unknown cell assigned to nearest robot.

    Args:
        robot_positions: {uav_id: (row, col)} for all robots.
        mask_state: (H, W) grid.

    Returns:
        {uav_id: (H, W) bool mask} — True where cell belongs to this robot.
    """
    h, w = mask_state.shape
    uav_ids = sorted(robot_positions.keys())

    if len(uav_ids) <= 1:
        full_mask = mask_state != 100
        return {uid: full_mask for uid in uav_ids}

    # Compute Manhattan distance from every cell to each robot
    rows, cols = np.mgrid[0:h, 0:w]
    min_dist = np.full((h, w), np.inf)
    owner = np.full((h, w), -1, dtype=np.int32)

    for uid in uav_ids:
        r, c = robot_positions[uid]
        dist = np.abs(rows - r) + np.abs(cols - c)
        dist_float = dist.astype(np.float64)
        # Tie-break: lower uav_id wins (add tiny offset)
        dist_float += uid * 1e-9
        closer = dist_float < min_dist
        min_dist[closer] = dist_float[closer]
        owner[closer] = uid

    result: Dict[int, np.ndarray] = {}
    for uid in uav_ids:
        result[uid] = owner == uid

    return result
