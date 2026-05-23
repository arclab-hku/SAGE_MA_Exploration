import math
from typing import Tuple


def compute_issue_distance(goal_xy: Tuple[float, float], odom_xy: Tuple[float, float]) -> float:
    dx = float(goal_xy[0] - odom_xy[0])
    dy = float(goal_xy[1] - odom_xy[1])
    return float(math.hypot(dx, dy))


def adaptive_arrival_tolerance(
    issued_dist_m: float,
    map_based_tol_m: float,
    max_pos_tol_m: float,
    min_pos_tol_m: float,
    adaptive_ratio: float,
) -> float:
    # Keep arrival criterion strict enough for planner handoff.
    # Do not use map-based lower bound here; otherwise tolerance may be too loose.
    base_tol = float(min_pos_tol_m)
    if issued_dist_m <= 1e-6:
        return float(max_pos_tol_m)
    return float(min(max_pos_tol_m, max(base_tol, adaptive_ratio * issued_dist_m)))


def maybe_keep_previous_goal(
    cur_rc: Tuple[int, int],
    candidate_rc: Tuple[int, int],
    prev_rc: Tuple[int, int],
    min_goal_delta_cells: int,
) -> Tuple[int, int]:
    cur_r, cur_c = cur_rc
    cand_r, cand_c = candidate_rc
    prev_r, prev_c = prev_rc

    cand_dist = abs(cand_r - cur_r) + abs(cand_c - cur_c)
    if cand_dist >= int(min_goal_delta_cells):
        return int(cand_r), int(cand_c)

    prev_dist = abs(prev_r - cur_r) + abs(prev_c - cur_c)
    if prev_dist >= int(min_goal_delta_cells):
        return int(prev_r), int(prev_c)

    return int(cand_r), int(cand_c)
