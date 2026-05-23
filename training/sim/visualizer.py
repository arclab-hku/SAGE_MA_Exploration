"""
Matplotlib side-by-side comparison visualizer.

Supports 2-way (baseline vs proposed) and 3-way (+ gbplanner) comparison.

Layout for 3-way: 3×3 grid
  [baseline grid]  [proposed grid]  [gbplanner grid]
  [coverage curve]  [comm volume]   [—]
  [collision]       [frontier exp]  [—]
"""
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .metrics import RunMetrics


def plot_comparison(
    baseline: RunMetrics,
    proposed: RunMetrics,
    baseline_maps: Optional[np.ndarray] = None,
    proposed_maps: Optional[np.ndarray] = None,
    output_path: str = "comparison.png",
    gbplanner: Optional[RunMetrics] = None,
    gbplanner_maps: Optional[np.ndarray] = None,
) -> str:
    """Generate comparison plot and save to file. Returns output path."""
    has_gb = gbplanner is not None
    ncols = 3 if has_gb else 2

    fig, axes = plt.subplots(3, ncols, figsize=(7 * ncols, 12))
    title = f"Baseline vs Proposed"
    if has_gb:
        title += " vs GBPlanner"
    title += f" — {baseline.map_name} map, {baseline.num_uavs} UAVs"
    fig.suptitle(title, fontsize=14)

    # Row 1: Grid snapshots
    if baseline_maps is not None:
        _plot_grid(axes[0, 0], baseline_maps, "Baseline (final)")
    else:
        axes[0, 0].text(0.5, 0.5, "No grid data", ha="center", va="center")
        axes[0, 0].set_title("Baseline (final)")

    if proposed_maps is not None:
        _plot_grid(axes[0, 1], proposed_maps, "Proposed (final)")
    else:
        axes[0, 1].text(0.5, 0.5, "No grid data", ha="center", va="center")
        axes[0, 1].set_title("Proposed (final)")

    if has_gb:
        if gbplanner_maps is not None:
            _plot_grid(axes[0, 2], gbplanner_maps, "GBPlanner (final)")
        else:
            axes[0, 2].text(0.5, 0.5, "No grid data", ha="center", va="center")
            axes[0, 2].set_title("GBPlanner (final)")

    # Row 2: Coverage curves + comm volume
    b_steps = [s.step for s in baseline.steps]
    p_steps = [s.step for s in proposed.steps]

    axes[1, 0].plot(b_steps, [s.coverage for s in baseline.steps], label="Baseline", color="blue")
    axes[1, 0].plot(p_steps, [s.coverage for s in proposed.steps], label="Proposed", color="red")
    if has_gb:
        g_steps = [s.step for s in gbplanner.steps]
        axes[1, 0].plot(g_steps, [s.coverage for s in gbplanner.steps], label="GBPlanner", color="green")
    axes[1, 0].axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% target")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Coverage")
    axes[1, 0].set_title("Coverage Over Time")
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].plot(b_steps, [s.comm_bytes_cumulative for s in baseline.steps], label="Baseline", color="blue")
    axes[1, 1].plot(p_steps, [s.comm_bytes_cumulative for s in proposed.steps], label="Proposed", color="red")
    if has_gb:
        g_steps = [s.step for s in gbplanner.steps]
        axes[1, 1].plot(g_steps, [s.comm_bytes_cumulative for s in gbplanner.steps], label="GBPlanner", color="green")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Cumulative Bytes")
    axes[1, 1].set_title("Communication Volume")
    axes[1, 1].legend()

    # Hide extra column cell if 3-way
    if has_gb:
        axes[1, 2].axis("off")

    # Row 3: Collision breakdown + frontier expirations
    axes[2, 0].plot(
        b_steps, [s.collisions_static for s in baseline.steps],
        label="Baseline static", color="blue",
    )
    axes[2, 0].plot(
        p_steps, [s.collisions_static for s in proposed.steps],
        label="Proposed static", color="red",
    )
    axes[2, 0].plot(
        p_steps, [s.collisions_inter_uav for s in proposed.steps],
        label="Proposed inter-UAV", color="orange", linestyle="--",
    )
    if has_gb:
        g_steps = [s.step for s in gbplanner.steps]
        axes[2, 0].plot(
            g_steps, [s.collisions_static for s in gbplanner.steps],
            label="GBPlanner static", color="green",
        )
        axes[2, 0].plot(
            g_steps, [s.collisions_inter_uav for s in gbplanner.steps],
            label="GBPlanner inter-UAV", color="lime", linestyle="--",
        )
    axes[2, 0].set_xlabel("Step")
    axes[2, 0].set_ylabel("Cumulative Collisions")
    axes[2, 0].set_title("Collision Breakdown")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(
        p_steps, [s.frontier_expirations for s in proposed.steps],
        label="Proposed", color="red",
    )
    axes[2, 1].plot(
        b_steps, [0] * len(b_steps),
        label="Baseline (0)", color="blue", alpha=0.3,
    )
    if has_gb:
        g_steps = [s.step for s in gbplanner.steps]
        axes[2, 1].plot(
            g_steps, [s.frontier_expirations for s in gbplanner.steps],
            label="GBPlanner", color="green",
        )
    axes[2, 1].set_xlabel("Step")
    axes[2, 1].set_ylabel("Cumulative Expirations")
    axes[2, 1].set_title("Frontier Expirations")
    axes[2, 1].legend()

    if has_gb:
        axes[2, 2].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_grid(ax, grid: np.ndarray, title: str) -> None:
    """Plot a 2D grid with color coding."""
    display = np.zeros((*grid.shape, 3), dtype=np.float32)
    display[grid == 0] = [1.0, 1.0, 1.0]     # free = white
    display[grid == 100] = [0.0, 0.0, 0.0]    # occupied = black
    display[grid == -1] = [0.5, 0.5, 0.5]     # unknown = gray
    ax.imshow(display, origin="upper")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def print_summary_table(
    baseline: RunMetrics,
    proposed: RunMetrics,
    gbplanner: Optional[RunMetrics] = None,
) -> str:
    """Return a formatted comparison table (2- or 3-way)."""
    bs = baseline.summary()
    ps = proposed.summary()
    gs = gbplanner.summary() if gbplanner else None
    has_gb = gs is not None

    if has_gb:
        header = f"{'Metric':<30} {'Baseline':>12} {'Proposed':>12} {'GBPlanner':>12} {'P/B':>8} {'G/B':>8}"
        sep = "-" * 86
    else:
        header = f"{'Metric':<30} {'Baseline':>15} {'Proposed':>15} {'Ratio':>10}"
        sep = "-" * 72

    lines = [header, sep]

    def _row(label, bv, pv, gv=None, fmt=".2f", ratio=True):
        bstr = f"{bv:{fmt}}" if isinstance(bv, float) else str(bv)
        pstr = f"{pv:{fmt}}" if isinstance(pv, float) else str(pv)

        def _ratio(num, denom):
            if (ratio and denom is not None and num is not None
                    and isinstance(denom, (int, float)) and isinstance(num, (int, float))
                    and denom != 0):
                return f"{num / denom:.2f}x"
            return "—"

        if has_gb:
            gstr = (f"{gv:{fmt}}" if isinstance(gv, float) else str(gv)) if gv is not None else "—"
            pb = _ratio(pv, bv)
            gb = _ratio(gv, bv) if gv is not None else "—"
            lines.append(f"{label:<30} {bstr:>12} {pstr:>12} {gstr:>12} {pb:>8} {gb:>8}")
        else:
            rstr = _ratio(pv, bv)
            lines.append(f"{label:<30} {bstr:>15} {pstr:>15} {rstr:>10}")

    _row("Final coverage", bs["final_coverage"], ps["final_coverage"],
         gs["final_coverage"] if gs else None)
    _row("Completion step", bs["completion_step"], ps["completion_step"],
         gs["completion_step"] if gs else None, fmt="d")
    _row("Success", bs["success"], ps["success"],
         gs["success"] if gs else None, ratio=False)
    _row("Total comm bytes", bs["total_comm_bytes"], ps["total_comm_bytes"],
         gs["total_comm_bytes"] if gs else None, fmt="d")
    _row("Comm bytes/step", bs["comm_bytes_per_step"], ps["comm_bytes_per_step"],
         gs["comm_bytes_per_step"] if gs else None)
    _row("Collision rate", bs["collision_rate"], ps["collision_rate"],
         gs["collision_rate"] if gs else None)
    _row("  Static collisions", bs["collisions_static"], ps["collisions_static"],
         gs["collisions_static"] if gs else None, fmt="d")
    _row("  Inter-UAV collisions", bs["collisions_inter_uav"], ps["collisions_inter_uav"],
         gs["collisions_inter_uav"] if gs else None, fmt="d")
    _row("Total replans", bs["total_replans"], ps["total_replans"],
         gs["total_replans"] if gs else None, fmt="d")
    _row("Frontier expirations", bs["total_frontier_expirations"], ps["total_frontier_expirations"],
         gs["total_frontier_expirations"] if gs else None, fmt="d")

    return "\n".join(lines)
