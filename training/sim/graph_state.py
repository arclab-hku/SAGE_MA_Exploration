"""
Exploration Graph (Layer 3): frontier/viewpoint/robot nodes + edges.

Node identity: spatial hash  "{type}_{x}_{y}"
Version: (lamport_counter, source_uav) — A1
Utility: locally derived from frontier in-edge count, never synced.
NavEdge pruning: budget-limited local neighborhood — A6
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import FrozenSet, Tuple

import numpy as np

from .config import GraphConfig, SensorConfig
from .sensor import ray_point


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

VersionT = Tuple[int, int]  # (lamport, source_uav)
_ZERO_VER: VersionT = (0, -1)


# ---------------------------------------------------------------------------
# Node / Edge data (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphNode:
    node_id: str              # "{type}_{x}_{y}"
    node_type: int            # 0=frontier, 1=viewpoint, 2=robot_self, 3=robot_other
    x: int                    # col in global frame
    y: int                    # row in global frame
    version: VersionT
    deleted: bool = False     # tombstone flag


@dataclass(frozen=True)
class GraphEdge:
    src_id: str
    dst_id: str
    edge_type: str            # "frontier_viewpoint", "viewpoint_robot", "nav"
    blocked: bool = False     # for nav edges: safety flag
    version: VersionT = _ZERO_VER


def make_node_id(node_type: int, x: int, y: int) -> str:
    """Spatial hash: quantized coordinate + type → deterministic identity."""
    return f"{node_type}_{x}_{y}"


# ---------------------------------------------------------------------------
# Exploration Graph (immutable operations — returns new objects)
# ---------------------------------------------------------------------------

class ExplorationGraph:
    """
    Mutable container but all public operations that modify state
    return description of changes (for delta production).
    Internal state is dict-based for O(1) lookup.
    """

    def __init__(self, source_uav: int):
        self._source = source_uav
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[tuple[str, str, str], GraphEdge] = {}
        self._tombstones: dict[str, int] = {}  # node_id → step_deleted
        self._lamport = 0

    @property
    def nodes(self) -> dict[str, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> dict[tuple[str, str, str], GraphEdge]:
        return self._edges

    @property
    def tombstones(self) -> dict[str, int]:
        return self._tombstones

    def _next_version(self) -> VersionT:
        self._lamport += 1
        return (self._lamport, self._source)

    def advance_lamport(self, remote_lamport: int) -> None:
        if remote_lamport > self._lamport:
            self._lamport = remote_lamport

    # -- frontier detection --

    def detect_frontiers(
        self,
        mask_state: np.ndarray,
        gcfg: GraphConfig | None = None,
    ) -> list[GraphNode]:
        """Detect frontier cells: unknown cell with >=1 free 4-neighbor.
        Vectorized with numpy for performance.
        Returns list of new/updated frontier nodes."""
        gcfg = gcfg or GraphConfig()
        h, w = mask_state.shape
        new_frontiers: list[GraphNode] = []
        active_frontier_ids: set[str] = set()

        # Vectorized frontier detection using numpy shifts
        is_unknown = mask_state == -1
        is_free = mask_state == 0

        # Check 4-neighbors for free cells (padded to handle borders)
        padded = np.pad(is_free, 1, mode="constant", constant_values=False)
        has_free_neighbor = (
            padded[:-2, 1:-1]  # up
            | padded[2:, 1:-1]  # down
            | padded[1:-1, :-2]  # left
            | padded[1:-1, 2:]  # right
        )

        frontier_mask = is_unknown & has_free_neighbor
        frontier_coords = np.argwhere(frontier_mask)  # (N, 2) array of [row, col]

        for rc in frontier_coords:
            r, c = int(rc[0]), int(rc[1])
            nid = make_node_id(gcfg.FRONTIER, c, r)
            active_frontier_ids.add(nid)

            if nid not in self._nodes:
                node = GraphNode(
                    node_id=nid,
                    node_type=gcfg.FRONTIER,
                    x=c, y=r,
                    version=self._next_version(),
                )
                self._nodes[nid] = node
                new_frontiers.append(node)

        # Remove stale frontiers
        stale = [
            nid for nid, n in self._nodes.items()
            if n.node_type == gcfg.FRONTIER and nid not in active_frontier_ids
        ]
        for nid in stale:
            self._delete_node(nid)

        return new_frontiers

    # -- viewpoint generation --

    def generate_viewpoints(
        self,
        robot_x: int,
        robot_y: int,
        mask_state: np.ndarray,
        scfg: SensorConfig | None = None,
        gcfg: GraphConfig | None = None,
    ) -> list[GraphNode]:
        """Generate viewpoints from 4 neighbors of robot position.
        Each viewpoint scores visible frontiers via ray cast."""
        scfg = scfg or SensorConfig()
        gcfg = gcfg or GraphConfig()
        h, w = mask_state.shape
        candidates = [
            (robot_x - 1, robot_y),
            (robot_x + 1, robot_y),
            (robot_x, robot_y - 1),
            (robot_x, robot_y + 1),
        ]

        frontier_positions = {
            (n.x, n.y)
            for n in self._nodes.values()
            if n.node_type == gcfg.FRONTIER and not n.deleted
        }

        new_viewpoints: list[GraphNode] = []

        for cx, cy in candidates:
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue
            if mask_state[cy, cx] == 100:
                continue

            visible_count = _count_visible_frontiers(
                cx, cy, frontier_positions, mask_state, scfg.view_range,
            )
            if visible_count < gcfg.viewpoint_min_visible_frontiers:
                continue

            nid = make_node_id(gcfg.VIEWPOINT, cx, cy)
            node = GraphNode(
                node_id=nid,
                node_type=gcfg.VIEWPOINT,
                x=cx, y=cy,
                version=self._next_version(),
            )
            self._nodes[nid] = node
            new_viewpoints.append(node)

            # Create edges: viewpoint → only visible frontiers within range
            for fx, fy in frontier_positions:
                if abs(fx - cx) > scfg.view_range or abs(fy - cy) > scfg.view_range:
                    continue
                if (fx - cx) ** 2 + (fy - cy) ** 2 > scfg.view_range ** 2:
                    continue
                fid = make_node_id(gcfg.FRONTIER, fx, fy)
                if fid in self._nodes:
                    eid = (nid, fid, "frontier_viewpoint")
                    self._edges[eid] = GraphEdge(
                        src_id=nid, dst_id=fid,
                        edge_type="frontier_viewpoint",
                        version=self._next_version(),
                    )

        return new_viewpoints

    # -- robot node --

    def update_robot_node(
        self,
        uav_id: int,
        x: int, y: int,
        is_self: bool = True,
        gcfg: GraphConfig | None = None,
    ) -> GraphNode:
        gcfg = gcfg or GraphConfig()
        ntype = gcfg.ROBOT_SELF if is_self else gcfg.ROBOT_OTHER
        nid = make_node_id(ntype, x, y)

        # Remove old robot node for this UAV
        old_ids = [
            n.node_id for n in self._nodes.values()
            if n.node_type == ntype
        ]
        for oid in old_ids:
            if oid != nid:
                self._delete_node(oid)

        node = GraphNode(
            node_id=nid, node_type=ntype,
            x=x, y=y,
            version=self._next_version(),
        )
        self._nodes[nid] = node

        # Edges: robot → nearby viewpoints (within 2*view_range)
        max_dist = 20  # limit edge fan-out
        for vid, vn in list(self._nodes.items()):
            if vn.node_type == gcfg.VIEWPOINT and not vn.deleted:
                if abs(vn.x - x) + abs(vn.y - y) <= max_dist:
                    eid = (nid, vid, "viewpoint_robot")
                    self._edges[eid] = GraphEdge(
                        src_id=nid, dst_id=vid,
                        edge_type="viewpoint_robot",
                        version=self._next_version(),
                    )
        return node

    # -- nav edges with budget-limited pruning (A6) --

    def add_nav_edge(
        self,
        src_x: int, src_y: int,
        dst_x: int, dst_y: int,
        mask_state: np.ndarray,
    ) -> GraphEdge | None:
        """Add a navigation edge if path is free. Check via ray cast."""
        h, w = mask_state.shape
        points = ray_point(src_x, src_y, dst_x, dst_y)
        blocked = False
        for px, py in points:
            if px < 0 or px >= w or py < 0 or py >= h:
                blocked = True
                break
            if mask_state[py, px] == 100:
                blocked = True
                break

        src_id = f"nav_{src_x}_{src_y}"
        dst_id = f"nav_{dst_x}_{dst_y}"
        eid = (src_id, dst_id, "nav")
        edge = GraphEdge(
            src_id=src_id, dst_id=dst_id,
            edge_type="nav",
            blocked=blocked,
            version=self._next_version(),
        )
        self._edges[eid] = edge
        return edge

    def prune_nav_edges(
        self,
        robot_x: int, robot_y: int,
        mask_state: np.ndarray,
        view_range: int = 10,
        gcfg: GraphConfig | None = None,
    ) -> int:
        """Budget-limited nav edge pruning in local neighborhood (A6).
        Returns number of edges pruned."""
        gcfg = gcfg or GraphConfig()
        radius = view_range * gcfg.nav_edge_prune_radius
        budget = gcfg.nav_edge_prune_budget
        pruned = 0

        nav_edges = [
            (eid, e) for eid, e in self._edges.items()
            if e.edge_type == "nav" and not e.blocked
        ]

        for eid, edge in nav_edges:
            if pruned >= budget:
                break
            # Parse coordinates from nav edge ids
            parts_src = edge.src_id.split("_")
            parts_dst = edge.dst_id.split("_")
            if len(parts_src) < 3 or len(parts_dst) < 3:
                continue
            sx, sy = int(parts_src[1]), int(parts_src[2])
            dx, dy = int(parts_dst[1]), int(parts_dst[2])

            # Only check edges in local neighborhood
            if (abs(sx - robot_x) > radius and abs(dx - robot_x) > radius):
                continue
            if (abs(sy - robot_y) > radius and abs(dy - robot_y) > radius):
                continue

            # Re-check if still free
            h, w = mask_state.shape
            points = ray_point(sx, sy, dx, dy)
            is_blocked = any(
                px < 0 or px >= w or py < 0 or py >= h
                or mask_state[py, px] == 100
                for px, py in points
            )
            if is_blocked:
                self._edges[eid] = GraphEdge(
                    src_id=edge.src_id, dst_id=edge.dst_id,
                    edge_type="nav", blocked=True,
                    version=self._next_version(),
                )
                pruned += 1

        return pruned

    # -- node deletion --

    def _delete_node(self, node_id: str, step: int = 0) -> None:
        if node_id in self._nodes:
            old = self._nodes[node_id]
            self._nodes[node_id] = GraphNode(
                node_id=old.node_id, node_type=old.node_type,
                x=old.x, y=old.y,
                version=self._next_version(),
                deleted=True,
            )
            self._tombstones[node_id] = step
        # Remove edges referencing this node
        to_remove = [
            eid for eid in self._edges
            if eid[0] == node_id or eid[1] == node_id
        ]
        for eid in to_remove:
            del self._edges[eid]

    def delete_node(self, node_id: str, step: int = 0) -> None:
        """Public deletion (used when UAV observes frontier is gone)."""
        self._delete_node(node_id, step)

    # -- tombstone GC --

    def gc_tombstones(self, current_step: int, ttl: int = 200) -> int:
        """Remove tombstones older than TTL. Returns count removed."""
        expired = [
            nid for nid, s in self._tombstones.items()
            if current_step - s > ttl
        ]
        for nid in expired:
            if nid in self._nodes:
                del self._nodes[nid]
            del self._tombstones[nid]
        return len(expired)

    # -- utility (locally derived from in-edge count, never synced) --

    def compute_utility(self, node_id: str) -> int:
        """Utility = number of frontier_viewpoint edges pointing to this node."""
        return sum(
            1 for eid, e in self._edges.items()
            if e.edge_type == "frontier_viewpoint" and eid[1] == node_id
        )

    # -- graph stats --

    def frontier_count(self) -> int:
        return sum(
            1 for n in self._nodes.values()
            if n.node_type == 0 and not n.deleted
        )

    def viewpoint_count(self) -> int:
        return sum(
            1 for n in self._nodes.values()
            if n.node_type == 1 and not n.deleted
        )

    def active_node_count(self) -> int:
        return sum(1 for n in self._nodes.values() if not n.deleted)

    def edge_count(self) -> int:
        return len(self._edges)

    def get_frontier_nodes(self) -> list[GraphNode]:
        return [
            n for n in self._nodes.values()
            if n.node_type == 0 and not n.deleted
        ]

    def get_viewpoint_nodes(self) -> list[GraphNode]:
        return [
            n for n in self._nodes.values()
            if n.node_type == 1 and not n.deleted
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_visible_frontiers(
    vx: int, vy: int,
    frontier_positions: set[tuple[int, int]],
    mask_state: np.ndarray,
    view_range: int,
) -> int:
    """Count frontiers visible from (vx, vy) within view_range via ray cast."""
    h, w = mask_state.shape
    count = 0
    for fx, fy in frontier_positions:
        if abs(fx - vx) > view_range or abs(fy - vy) > view_range:
            continue
        if (fx - vx) ** 2 + (fy - vy) ** 2 > view_range ** 2:
            continue
        # Ray trace to check occlusion
        points = ray_point(vx, vy, fx, fy)
        occluded = False
        for px, py in points[1:-1]:  # skip start and end
            if px < 0 or px >= w or py < 0 or py >= h:
                occluded = True
                break
            if mask_state[py, px] == 100:
                occluded = True
                break
        if not occluded:
            count += 1
    return count
