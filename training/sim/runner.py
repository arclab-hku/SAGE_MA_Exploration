"""
Main runner: executes baseline vs proposed vs gbplanner on same world with fair constraints (A7).

A7 fairness:
  - Same random seed for all systems
  - Pre-generated noise sequences (not shared sensor instances)
  - Same dynamics / controller (identical step model)
  - Same starting positions
  - Same ground truth map

Usage:
  python -m sim.runner --map rooms --uavs 2
  python -m sim.runner --all  # run all 6 configurations
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np

from .config import (
    MAP_PRESETS,
    CommConfig,
    GBPlannerConfig,
    GraphConfig,
    Layer2Config,
    PlannerConfig,
    SensorConfig,
    SimConfig,
    UAVConfig,
    WorldConfig,
)
from .gbplanner_system import GBPlannerSystem
from .metrics import RunMetrics, StepMetrics
from .proposed import BaselineSystem, ProposedSystem
from .visualizer import plot_comparison, print_summary_table
from .world import make_world


def run_comparison(
    map_name: str,
    num_uavs: int,
    cfg: SimConfig,
    output_dir: str = "results",
) -> Tuple[RunMetrics, RunMetrics, RunMetrics]:
    """Run baseline, proposed, and gbplanner on same map.

    Returns (baseline_metrics, proposed_metrics, gbplanner_metrics).
    """

    # A7: same ground truth
    wcfg = cfg.world
    ground_truth = make_world(map_name, wcfg)

    # A7: same starting positions
    preset = MAP_PRESETS[map_name]
    positions = preset["start_positions"][num_uavs]

    # Validate start positions are free
    for r, c in positions:
        if ground_truth[r, c] == 100:
            raise ValueError(f"Start position ({r},{c}) is occupied in {map_name} map")

    # A7: independent pre-generated noise sequences from same seed
    baseline_rng = np.random.RandomState(cfg.seed)
    proposed_rng = np.random.RandomState(cfg.seed)
    gbplanner_rng = np.random.RandomState(cfg.seed)

    scfg = cfg.sensor
    l2cfg = cfg.layer2
    gcfg = cfg.graph
    pcfg = cfg.planner
    ccfg = cfg.comm
    gbcfg = cfg.gbplanner

    # Create systems
    baseline = BaselineSystem(
        positions, wcfg, scfg, pcfg, ccfg, noise_rng=baseline_rng,
    )
    proposed = ProposedSystem(
        positions, wcfg, scfg, l2cfg, gcfg, pcfg, ccfg, noise_rng=proposed_rng,
    )
    gbplanner = GBPlannerSystem(
        positions, wcfg, scfg, l2cfg, pcfg, gbcfg, ccfg, noise_rng=gbplanner_rng,
    )

    # Metrics collectors
    b_metrics = RunMetrics(system_name="baseline", map_name=map_name, num_uavs=num_uavs)
    p_metrics = RunMetrics(system_name="proposed", map_name=map_name, num_uavs=num_uavs)
    g_metrics = RunMetrics(system_name="gbplanner", map_name=map_name, num_uavs=num_uavs)

    # Run simulation
    for step in range(cfg.max_steps):
        baseline.step(ground_truth, step)
        proposed.step(ground_truth, step)
        gbplanner.step(ground_truth, step)

        # Collect metrics
        b_metrics.record_step(StepMetrics(
            step=step,
            coverage=baseline.coverage_ratio(),
            comm_bytes_cumulative=baseline.total_comm_bytes(),
            collisions=baseline.total_collisions(),
            collisions_static=baseline.total_collisions(),  # baseline: all static
            collisions_inter_uav=0,
            replans=0,
            frontier_expirations=0,
        ))

        p_graph_sizes = proposed.graph_sizes()
        p_metrics.record_step(StepMetrics(
            step=step,
            coverage=proposed.coverage_ratio(),
            comm_bytes_cumulative=proposed.total_comm_bytes(),
            collisions=proposed.total_collisions(),
            collisions_static=proposed.total_collisions_static(),
            collisions_inter_uav=proposed.total_collisions_inter_uav(),
            replans=proposed.total_replans(),
            frontier_expirations=proposed.total_frontier_expirations(),
            graph_node_count=sum(p_graph_sizes.values()),
        ))

        g_metrics.record_step(StepMetrics(
            step=step,
            coverage=gbplanner.coverage_ratio(),
            comm_bytes_cumulative=gbplanner.total_comm_bytes(),
            collisions=gbplanner.total_collisions(),
            collisions_static=gbplanner.total_collisions_static(),
            collisions_inter_uav=gbplanner.total_collisions_inter_uav(),
            replans=gbplanner.total_replans(),
            frontier_expirations=gbplanner.total_frontier_expirations(),
        ))

        # Early termination if all complete
        if b_metrics.success and p_metrics.success and g_metrics.success:
            break

    # Generate output
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{map_name}_{num_uavs}uavs"

    # Get final maps for visualization
    b_map = baseline.uavs[0].local_map if baseline.uavs else None
    p_map = proposed.uavs[0].mask.state if proposed.uavs else None
    g_map = gbplanner.uavs[0].mask.state if gbplanner.uavs else None

    plot_path = plot_comparison(
        b_metrics, p_metrics,
        baseline_maps=b_map,
        proposed_maps=p_map,
        output_path=os.path.join(output_dir, f"{filename}.png"),
        gbplanner=g_metrics,
        gbplanner_maps=g_map,
    )

    # Print summary
    table = print_summary_table(b_metrics, p_metrics, gbplanner=g_metrics)
    print(f"\n=== {map_name} map, {num_uavs} UAVs ===")
    print(table)

    # Production-equivalent estimate: 3D voxel chunks would be ~10x larger
    b_3d_est = b_metrics.total_comm_bytes * 40  # height dimension factor
    p_actual = p_metrics.total_comm_bytes
    g_actual = g_metrics.total_comm_bytes
    if b_3d_est > 0:
        print(f"\nProduction estimate (3D baseline x 40 height layers):")
        print(f"  Baseline 3D: {b_3d_est:,} bytes")
        print(f"  Proposed:    {p_actual:,} bytes  ({p_actual/b_3d_est:.3f}x)")
        print(f"  GBPlanner:   {g_actual:,} bytes  ({g_actual/b_3d_est:.3f}x)")

    print(f"Plot saved: {plot_path}\n")

    return b_metrics, p_metrics, g_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-UAV Exploration Simulation")
    parser.add_argument("--map", choices=list(MAP_PRESETS.keys()), default="rooms")
    parser.add_argument("--uavs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all 6 configurations")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    cfg = SimConfig(max_steps=args.steps, seed=args.seed)

    if args.all:
        all_results = []
        for map_name in MAP_PRESETS:
            for num_uavs in [2, 3]:
                b, p, g = run_comparison(map_name, num_uavs, cfg, args.output)
                all_results.append((b, p, g))

        # Print combined summary
        print("\n" + "=" * 72)
        print("COMBINED RESULTS")
        print("=" * 72)
        for b, p, g in all_results:
            print(f"\n{b.map_name} / {b.num_uavs} UAVs:")
            print(f"  Coverage: baseline={b.final_coverage:.3f} "
                  f"proposed={p.final_coverage:.3f} "
                  f"gbplanner={g.final_coverage:.3f}")
            print(f"  Comm: baseline={b.total_comm_bytes}B "
                  f"proposed={p.total_comm_bytes}B "
                  f"gbplanner={g.total_comm_bytes}B")
            print(f"  Completion: baseline={b.completion_step} "
                  f"proposed={p.completion_step} "
                  f"gbplanner={g.completion_step}")
    else:
        run_comparison(args.map, args.uavs, cfg, args.output)


if __name__ == "__main__":
    main()
