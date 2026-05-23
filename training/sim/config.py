"""
Simulation configuration — all tunable parameters in one place.

Engineering requirements addressed:
  1. Global coordinate frame (unified origin, no per-UAV transform)
  2. Layer2Delta uses linear cell_id (no uint8 overflow on maps > 255)
  3. Confidence / decay for occupied cells (observation count + time decay)
  4. Explicit planning switch thresholds in meters (Layer2 → Layer1, fallback)
  5. Extended metrics (success, collision, replan, frontier expiry, completion)

Adjustments (user review):
  A1. Version = (lamport_counter, source_uav) — see crdt.py
  A2. Digest + snapshot pull for delta reliability — see comm.py
  A3. Layer2 merge uses obs_count, not just state priority — see explored_mask.py
  A4. Thresholds in meters, converted to cells via resolution
  A5. Fallback = BFS toward nearest unknown, not random walk
  A6. NavEdge pruning budget = K edges/step in local neighborhood
  A7. Fair comparison: shared seed, sensor, dynamics — see runner.py
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldConfig:
    width: int = 80
    height: int = 80
    resolution: float = 0.3    # meters per cell (A4)
    # Cell states in ground truth
    FREE: int = 0
    OCCUPIED: int = 100
    UNKNOWN: int = -1


@dataclass(frozen=True)
class SensorConfig:
    view_range: int = 10          # radius in cells
    num_rays: int = 72            # 360 / 5 degrees


@dataclass(frozen=True)
class Layer2Config:
    """Global 2D coarse grid — 3-state with confidence (A3)."""
    UNKNOWN: int = -1
    FREE: int = 0
    OCCUPIED: int = 100

    # Confidence / decay
    max_observation_count: int = 10
    decay_interval: int = 50          # steps between decay ticks
    decay_amount: int = 0             # disabled for static maps (no false-obstacle risk)
    clear_threshold: int = -1         # never revert to unknown in static environments

    # Merge: higher obs_count wins when states conflict (A3)
    # occupied(count=1) does NOT override free(count=8)
    merge_min_obs_to_override: int = 3  # remote needs >= this obs_count to override

    # Delta compression
    min_delta_cells: int = 1             # flush immediately (obstacle propagation is critical)
    max_delta_defer_steps: int = 1       # never defer


@dataclass(frozen=True)
class GraphConfig:
    # Node types (matches production encoding)
    FRONTIER: int = 0
    VIEWPOINT: int = 1
    ROBOT_SELF: int = 2
    ROBOT_OTHER: int = 3

    frontier_min_unknown_neighbors: int = 1
    viewpoint_min_visible_frontiers: int = 1

    # NavEdge pruning budget (A6)
    nav_edge_prune_budget: int = 8       # max edges to verify per step
    nav_edge_prune_radius: int = 2       # multiplier of view_range for local scope


@dataclass(frozen=True)
class CRDTConfig:
    tombstone_ttl: int = 200
    # Version = (lamport, source_uav) (A1)
    # DELETE wins at same version
    # NavEdge: blocked=True wins (safety bias)


@dataclass(frozen=True)
class CommConfig:
    latency_steps: int = 1
    loss_rate: float = 0.0              # 0.0 = lossless; >0 triggers digest path (A2)

    # Digest / snapshot (A2)
    digest_interval: int = 20           # steps between periodic state digest
    snapshot_max_cells: int = 500       # max cells per snapshot response

    # Byte costs
    chunk_bytes_per_voxel: int = 5      # baseline: uint32 addr + uint8 occ
    chunk_size: int = 200               # voxels per chunk (production)
    graph_node_bytes: int = 16          # id + type + version(lamport,src) + coords
    graph_edge_bytes: int = 8
    layer2_cell_bytes: int = 6          # cell_id(uint16) + state(int8) + obs_count(uint8) + version(uint16)
    digest_bytes: int = 32              # hash + seq


@dataclass(frozen=True)
class PlannerConfig:
    # Thresholds in meters (A4), converted at runtime via resolution
    layer2_switch_m: float = 4.5        # > this → Layer2 A*
    layer1_switch_m: float = 2.4        # < this → Layer1 fine planning
    hysteresis_m: float = 0.9           # dead band to prevent oscillation
    max_replan_per_step: int = 2
    # Fallback: BFS toward nearest unknown in local free area (A5)
    fallback_bfs_radius: int = 20       # cells to search in fallback BFS

    # Frontier scoring
    utility_weight: float = 1.0
    distance_weight: float = 1.0
    staleness_penalty: float = 0.0  # disabled for fair comparison

    # A* cost for unknown cells (1.0 = same as free, same as baseline)
    unknown_cell_cost: float = 1.0

    # Corridor commit disabled (threshold 0 = never activate)
    corridor_max_free_neighbors: int = 0

    def layer2_switch_cells(self, resolution: float) -> int:
        return int(self.layer2_switch_m / resolution)

    def layer1_switch_cells(self, resolution: float) -> int:
        return int(self.layer1_switch_m / resolution)

    def hysteresis_cells(self, resolution: float) -> int:
        return max(1, int(self.hysteresis_m / resolution))


@dataclass(frozen=True)
class GBPlannerConfig:
    """GBPlanner-style RRG exploration parameters."""
    num_vertices_max: int = 200        # RRG node cap
    num_samples_per_step: int = 50     # samples per RRG rebuild
    k_nearest: int = 8                 # RRG connection count
    max_edge_length: int = 15          # max edge length in cells
    rebuild_interval: int = 5          # steps between RRG rebuilds
    gain_weight: float = 1.0           # gain scaling in score
    min_gain_threshold: float = 1.0    # minimum gain to consider a node


@dataclass(frozen=True)
class UAVConfig:
    num_uavs: int = 2


@dataclass(frozen=True)
class MetricsConfig:
    """Extended metrics (req #5)."""
    track_coverage: bool = True
    track_comm_bytes: bool = True
    track_stale_frontiers: bool = True
    track_graph_size: bool = True
    track_collision_rate: bool = True
    track_replan_count: bool = True
    track_frontier_expiry_rate: bool = True
    track_task_completion_time: bool = True
    track_success_rate: bool = True


@dataclass(frozen=True)
class SimConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    layer2: Layer2Config = field(default_factory=Layer2Config)
    graph: GraphConfig = field(default_factory=GraphConfig)
    crdt: CRDTConfig = field(default_factory=CRDTConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    gbplanner: GBPlannerConfig = field(default_factory=GBPlannerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    max_steps: int = 500
    seed: int = 42


MAP_PRESETS: dict[str, dict] = {
    "rooms": {
        "description": "4 rooms connected by corridors",
        "start_positions": {
            2: [(10, 10), (70, 70)],
            3: [(10, 10), (70, 70), (10, 70)],
            4: [(10, 10), (70, 70), (10, 70), (70, 10)],
        },
    },
    "maze": {
        "description": "Maze-like corridors with dead ends",
        "start_positions": {
            2: [(1, 1), (78, 78)],
            3: [(1, 1), (78, 78), (1, 78)],
            4: [(1, 1), (78, 78), (1, 78), (78, 1)],
        },
    },
    "open": {
        "description": "Large open area with scattered obstacles",
        "start_positions": {
            2: [(20, 20), (60, 60)],
            3: [(20, 20), (60, 60), (40, 40)],
            4: [(20, 20), (60, 60), (20, 60), (60, 20)],
        },
    },
}
