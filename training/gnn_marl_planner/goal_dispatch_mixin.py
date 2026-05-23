from __future__ import annotations

import math
import time
import numpy as np
from .goal_logic import adaptive_arrival_tolerance, compute_issue_distance
from .planner_metrics import update_ema, l2_distance_xy
from .uav_index import uav_key_from_index

class GoalDispatchMixin:
    def _segment_completion_state(self, drone_id: int, uav_key: str, goal_grid):
        if uav_key not in self.uav_ros_odoms:
            return None
        if drone_id >= len(self.graph_pose_committed_ds):
            return None
        start_grid = self.graph_pose_committed_ds[drone_id]
        if start_grid is None:
            return None
        start_grid = (int(start_grid[0]), int(start_grid[1]))
        goal_grid = (int(goal_grid[0]), int(goal_grid[1]))
        if start_grid == goal_grid:
            return None

        start_xy = self.pose_in_ros(
            [int(start_grid[0]), int(start_grid[1])], self.resolution, self.downsample_factor
        )
        goal_xy = self.pose_in_ros(
            [int(goal_grid[0]), int(goal_grid[1])], self.resolution, self.downsample_factor
        )
        odom_xy = self.uav_ros_odoms[uav_key]

        seg_x = float(goal_xy[0] - start_xy[0])
        seg_y = float(goal_xy[1] - start_xy[1])
        seg_len_sq = seg_x * seg_x + seg_y * seg_y
        if seg_len_sq <= 1e-6:
            return None
        seg_len = math.sqrt(seg_len_sq)

        rel_x = float(odom_xy[0] - start_xy[0])
        rel_y = float(odom_xy[1] - start_xy[1])
        progress_ratio = (rel_x * seg_x + rel_y * seg_y) / seg_len_sq
        cross_track_m = abs(rel_x * seg_y - rel_y * seg_x) / seg_len
        return {
            "start_grid": [int(start_grid[0]), int(start_grid[1])],
            "goal_grid": [int(goal_grid[0]), int(goal_grid[1])],
            "progress_ratio": float(progress_ratio),
            "cross_track_m": float(cross_track_m),
        }

    def _indices_to_viewpoint_coords(self, drone_id: int, indices, limit: int = 16):
        coords = []
        vp_items = self._ordered_viewpoint_items(drone_id)
        for raw_idx in list(indices)[:limit]:
            try:
                vp_idx = int(raw_idx) - int(self.frontier_num)
            except Exception:
                continue
            if 0 <= vp_idx < len(vp_items):
                value = vp_items[vp_idx][1]
                coords.append([int(value[0]), int(value[1])])
        return coords

    def _coord_to_graph_index(self, drone_id: int, coord):
        target = (int(coord[0]), int(coord[1]))
        vp_items = self._ordered_viewpoint_items(drone_id)
        for vp_idx, (_, value) in enumerate(vp_items):
            if (int(value[0]), int(value[1])) == target:
                return int(self.frontier_num) + int(vp_idx)
        return None

    def _resolve_execution_anchor(self, goal_rc, anchor_rc=None):
        goal_r, goal_c = int(goal_rc[0]), int(goal_rc[1])
        reason = "center"
        if not getattr(self, "execution_anchor_fallback_enabled", False):
            return [goal_r, goal_c], False, reason
        data = getattr(self, "downsampled_map_data", None)
        if data is None or getattr(data, "size", 0) == 0:
            return [goal_r, goal_c], False, "no_map"
        if not (0 <= goal_r < data.shape[0] and 0 <= goal_c < data.shape[1]):
            return [goal_r, goal_c], False, "goal_oob"
        if int(data[goal_r, goal_c]) != 100:
            return [goal_r, goal_c], False, reason

        max_radius = int(getattr(self, "execution_anchor_search_radius_cells", 0))
        if max_radius <= 0:
            return [goal_r, goal_c], False, "center_occupied"

        candidates = []
        for radius in range(1, max_radius + 1):
            r0 = max(0, goal_r - radius)
            r1 = min(data.shape[0], goal_r + radius + 1)
            c0 = max(0, goal_c - radius)
            c1 = min(data.shape[1], goal_c + radius + 1)
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if max(abs(rr - goal_r), abs(cc - goal_c)) != radius:
                        continue
                    if int(data[rr, cc]) != 0:
                        continue
                    anchor_dist = 0
                    if anchor_rc is not None:
                        anchor_dist = abs(int(anchor_rc[0]) - rr) + abs(int(anchor_rc[1]) - cc)
                    goal_dist = abs(rr - goal_r) + abs(cc - goal_c)
                    candidates.append((goal_dist, anchor_dist, rr, cc))
            if candidates:
                break
        if not candidates:
            return [goal_r, goal_c], False, "center_occupied"
        candidates.sort()
        _, _, rr, cc = candidates[0]
        return [int(rr), int(cc)], True, "center_occupied_fallback"

    def _choose_alternative_neighbor_viewpoint(
        self,
        drone_id: int,
        uav_key: str,
        neighbor_indices,
        claimed_viewpoints,
        banned_coords,
        anchor_rc,
    ):
        vp_items = self._ordered_viewpoint_items(drone_id)
        if not vp_items:
            return None
        drone_pos = self.uav_odoms.get(uav_key, anchor_rc if anchor_rc is not None else (0, 0))
        seen = set()
        candidates = []
        for raw_idx in neighbor_indices:
            try:
                vp_idx = int(raw_idx) - int(self.frontier_num)
            except Exception:
                continue
            if not (0 <= vp_idx < len(vp_items)):
                continue
            _, value = vp_items[vp_idx]
            coord = (int(value[0]), int(value[1]))
            if coord in seen or coord in claimed_viewpoints or coord in banned_coords:
                continue
            seen.add(coord)
            utility = float(value[2]) if len(value) > 2 else 0.0
            if anchor_rc is not None:
                dist_anchor = abs(coord[0] - int(anchor_rc[0])) + abs(coord[1] - int(anchor_rc[1]))
            else:
                dist_anchor = 0
            dist_drone = abs(coord[0] - int(drone_pos[0])) + abs(coord[1] - int(drone_pos[1]))
            candidates.append((-utility, dist_anchor, dist_drone, coord[0], coord[1]))
        if not candidates:
            return None
        candidates.sort()
        _, _, _, vx, vy = candidates[0]
        return int(vx), int(vy)

    def _should_refresh_goal(self, idx: int, uav_key: str, prev_goal_x: int, prev_goal_y: int) -> bool:
        # Primary path: confirmed arrival.
        if self.arrive_flag[idx]:
            return True

        # Mechanism upgrade: near-goal prefetch to avoid "late dispatch".
        if not getattr(self, "goal_prefetch_enabled", False):
            return False
        if uav_key not in self.uav_ros_odoms:
            return False
        if idx >= len(self.goal_issue_ts):
            return False
        issue_ts = float(self.goal_issue_ts[idx])
        if issue_ts <= 0.0:
            return False
        age_s = time.monotonic() - issue_ts
        if age_s < float(self.goal_prefetch_min_age_s):
            return False

        goal_xy = self.pose_in_ros(
            [int(prev_goal_x), int(prev_goal_y)], self.resolution, self.downsample_factor
        )
        odom_xy = self.uav_ros_odoms[uav_key]
        dist_m = float(math.hypot(goal_xy[0] - odom_xy[0], goal_xy[1] - odom_xy[1]))
        return dist_m <= float(self.goal_prefetch_dist_m)

    def send_ros_goal(self, drone_id):
        # 发布uav在ros中要去的位置
        if drone_id >= len(self.goal_of_drone_msg):
            return
        msg = self.goal_of_drone_msg[drone_id]
        if msg is False:
            return
        if drone_id < len(self.goal_of_drone):
            goal_xy = self.goal_of_drone[drone_id]
            prev_goal = self.last_published_goal[drone_id]
            if (
                self.suppress_duplicate_goal_publish
                and prev_goal is not None
                and l2_distance_xy(goal_xy, prev_goal) <= self.duplicate_goal_eps_m
            ):
                self._metric_goal_publish_dup_skip += 1
                return
        self.publisher_list[drone_id].publish(msg)
        self._metric_goal_publish_total += 1
        if drone_id < len(self.goal_of_drone):
            uav_key = uav_key_from_index(self.uav_odoms, drone_id)
            self.last_published_goal[drone_id] = self.goal_of_drone[drone_id]
            self.goal_issue_ts[drone_id] = time.monotonic()
            if uav_key in self.uav_ros_odoms:
                self.goal_issue_distance_m[drone_id] = compute_issue_distance(
                    self.goal_of_drone[drone_id], self.uav_ros_odoms[uav_key]
                )
            else:
                self.goal_issue_distance_m[drone_id] = 0.0

    def send_each_ros_goal(self):
        # 发布uav在ros中要去的位置
        for drone_id in range(self.drone_num):
            self.send_ros_goal(drone_id)

    def _resolve_hold_goal(self, drone_id: int, uav_key: str):
        hold_r, hold_c = 0, 0
        if drone_id < len(self.uav_goal_grid_map):
            hold_r = int(self.uav_goal_grid_map[drone_id][0])
            hold_c = int(self.uav_goal_grid_map[drone_id][1])
        if uav_key in self.uav_odoms:
            hold_r = int(self.uav_odoms[uav_key][0])
            hold_c = int(self.uav_odoms[uav_key][1])

        if uav_key in self.uav_ros_odoms:
            hold_goal_xy = [
                float(self.uav_ros_odoms[uav_key][0]),
                float(self.uav_ros_odoms[uav_key][1]),
            ]
        else:
            hold_goal_xy = self.pose_in_ros(
                [int(hold_r), int(hold_c)], self.resolution, self.downsample_factor
            )
        return (int(hold_r), int(hold_c)), hold_goal_xy

    def _set_goal_slot_inactive(self, drone_id: int, hold_rc, hold_goal_xy):
        if drone_id < len(self.goal_of_drone_msg):
            self.goal_of_drone_msg[drone_id] = False
        if drone_id < len(self.goal_of_drone):
            self.goal_of_drone[drone_id] = list(hold_goal_xy)
        if drone_id < len(self.goal_of_drone_grid):
            self.goal_of_drone_grid[drone_id] = [int(hold_rc[0]), int(hold_rc[1])]
        if drone_id < len(self.goal_exec_grid):
            self.goal_exec_grid[drone_id] = [int(hold_rc[0]), int(hold_rc[1])]
        if drone_id < len(self.uav_goal_grid_map):
            self.uav_goal_grid_map[drone_id][0] = int(hold_rc[0])
            self.uav_goal_grid_map[drone_id][1] = int(hold_rc[1])
        if drone_id < len(self.goal_grid_initialized):
            self.goal_grid_initialized[drone_id] = True
        if drone_id < len(self.goal_issue_ts):
            self.goal_issue_ts[drone_id] = 0.0
        if drone_id < len(self.goal_issue_distance_m):
            self.goal_issue_distance_m[drone_id] = 0.0
        if drone_id < len(self.goal_issue_grid):
            self.goal_issue_grid[drone_id] = None
        if drone_id < len(self.goal_issue_exec_grid):
            self.goal_issue_exec_grid[drone_id] = None
        if drone_id < len(self.goal_issue_exec_reason):
            self.goal_issue_exec_reason[drone_id] = None
        if drone_id < len(self.goal_issue_odom_grid):
            self.goal_issue_odom_grid[drone_id] = None
        if drone_id < len(self.goal_issue_odom_xy):
            self.goal_issue_odom_xy[drone_id] = None
        if drone_id < len(self.goal_last_dist_m):
            self.goal_last_dist_m[drone_id] = float("inf")
        if drone_id < len(self.goal_stall_counts):
            self.goal_stall_counts[drone_id] = 0
        if drone_id < len(self.arrive_flag):
            self.arrive_flag[drone_id] = True
        if drone_id < len(self.arrive_stable_counts):
            self.arrive_stable_counts[drone_id] = 0
        if drone_id < len(self.last_published_goal):
            self.last_published_goal[drone_id] = None
        if drone_id < len(self.active_goal_valid):
            self.active_goal_valid[drone_id] = False
        if (
            drone_id < len(self.graph_pose_committed_ds)
            and drone_id < len(self.graph_pose_ds)
        ):
            self.graph_pose_ds[drone_id] = list(self.graph_pose_committed_ds[drone_id])
            self.graph_pose_initialized[drone_id] = True
        if drone_id < len(self.active_goal_ds) and drone_id < len(self.graph_pose_committed_ds):
            self.active_goal_ds[drone_id] = list(self.graph_pose_committed_ds[drone_id])

    def _commit_drone_graph_pose(self, drone_id: int, reason: str):
        target_rc = None
        if drone_id < len(self.active_goal_valid) and self.active_goal_valid[drone_id]:
            target_rc = self.active_goal_ds[drone_id]
        elif drone_id < len(self.goal_of_drone_grid):
            target_rc = self.goal_of_drone_grid[drone_id]
        elif drone_id < len(self.graph_pose_ds):
            target_rc = self.graph_pose_ds[drone_id]
        if target_rc is None:
            return
        committed = [int(target_rc[0]), int(target_rc[1])]
        if drone_id < len(self.graph_pose_committed_ds):
            self.graph_pose_committed_ds[drone_id] = list(committed)
        if drone_id < len(self.graph_pose_ds):
            self.graph_pose_ds[drone_id] = list(committed)
        if drone_id < len(self.graph_pose_initialized):
            self.graph_pose_initialized[drone_id] = True
        if drone_id < len(self.active_goal_valid):
            self.active_goal_valid[drone_id] = False
        self._append_event_log(
            "graph_pose_commit",
            drone_id=int(drone_id),
            reason=str(reason),
            committed_grid=[int(committed[0]), int(committed[1])],
        )
        self._mark_viewpoint_consumed(committed, drone_id=drone_id, reason=reason)

    def _mark_viewpoint_consumed(self, coord, drone_id: int, reason: str):
        if coord is None:
            return False
        consumed_key = (int(coord[0]), int(coord[1]))
        if consumed_key in self._consumed_viewpoints_global:
            return False
        self._consumed_viewpoints_global.add(consumed_key)
        self._append_event_log(
            "consumed_viewpoint_global",
            drone_id=int(drone_id),
            reason=str(reason),
            consumed_grid=[int(consumed_key[0]), int(consumed_key[1])],
            total_consumed=int(len(self._consumed_viewpoints_global)),
        )
        return True

    def _freeze_drone_goal(self, drone_id: int, reason: str, publish_hold: bool, append_slot: bool):
        uav_key = uav_key_from_index(self.uav_odoms, drone_id) or str(drone_id)
        hold_rc, hold_goal_xy = self._resolve_hold_goal(drone_id, uav_key)

        if publish_hold and drone_id < len(self.publisher_list):
            msg = self.To_goal_msgs(hold_goal_xy)
            self.publisher_list[drone_id].publish(msg)
            self.get_logger().warn(
                f"freeze goal on drone {drone_id}: reason={reason}, "
                f"hold_grid=({hold_rc[0]},{hold_rc[1]}), "
                f"hold_xy=({hold_goal_xy[0]:.2f},{hold_goal_xy[1]:.2f})"
            )
        self._append_event_log(
            "freeze_goal",
            drone_id=int(drone_id),
            reason=str(reason),
            hold_grid=[int(hold_rc[0]), int(hold_rc[1])],
            hold_xy=[float(hold_goal_xy[0]), float(hold_goal_xy[1])],
            publish_hold=bool(publish_hold),
        )

        if append_slot:
            self.goal_of_drone_msg.append(False)
            self.goal_of_drone.append(list(hold_goal_xy))
            self.goal_of_drone_grid.append([int(hold_rc[0]), int(hold_rc[1])])
        self._set_goal_slot_inactive(drone_id, hold_rc, hold_goal_xy)
        if (
            getattr(self, "manual_step_mode", False)
            and reason == "arrival_timeout"
            and drone_id < len(self.graph_pose_ds)
        ):
            self.graph_pose_ds[drone_id] = [int(hold_rc[0]), int(hold_rc[1])]
            if drone_id < len(self.graph_pose_initialized):
                self.graph_pose_initialized[drone_id] = True
            if drone_id < len(self.active_goal_ds):
                self.active_goal_ds[drone_id] = [int(hold_rc[0]), int(hold_rc[1])]
        return hold_rc, hold_goal_xy

    def _freeze_all_goals(self, reason: str):
        if len(self.goal_of_drone_msg) < self.drone_num:
            self.goal_of_drone_msg = [False for _ in range(self.drone_num)]
            self.goal_of_drone = [[0.0, 0.0] for _ in range(self.drone_num)]
            self.goal_of_drone_grid = [[0, 0] for _ in range(self.drone_num)]
        if len(self.goal_exec_grid) < self.drone_num:
            self.goal_exec_grid = [[0, 0] for _ in range(self.drone_num)]
        for drone_id in range(self.drone_num):
            should_publish_hold = (
                drone_id < len(self.last_published_goal)
                and self.last_published_goal[drone_id] is not None
            )
            self._freeze_drone_goal(
                drone_id=drone_id,
                reason=reason,
                publish_hold=should_publish_hold,
                append_slot=False,
            )

    def get_goal(self):
        self.goal_of_drone_msg = []
        self.goal_of_drone = []
        self.goal_of_drone_grid = []
        temp_delete_viewpoint = []

        refresh_plan = []
        claimed_viewpoints = set()
        for i, _action in enumerate(self.action_list):
            prev_goal_x = int(self.uav_goal_grid_map[i][0])
            prev_goal_y = int(self.uav_goal_grid_map[i][1])
            uav_key = uav_key_from_index(self.uav_odoms, i) or str(i)
            refresh = self._should_refresh_goal(i, uav_key, prev_goal_x, prev_goal_y)
            refresh_plan.append((refresh, uav_key, prev_goal_x, prev_goal_y))
            if not refresh:
                claimed_viewpoints.add((prev_goal_x, prev_goal_y))

        for i, action in enumerate(self.action_list):
            refresh, uav_key, prev_goal_x, prev_goal_y = refresh_plan[i]
            if refresh:
                direction = int(action[0][0])
                odom_rc = self.uav_odoms.get(uav_key, (None, None))
                anchor_rc = None
                if hasattr(self, "_effective_pos_for_graph"):
                    try:
                        eff_r, eff_c = self._effective_pos_for_graph(i, uav_key)
                        anchor_rc = [int(eff_r), int(eff_c)]
                    except Exception:
                        anchor_rc = None
                full_indices = []
                if i < len(getattr(self, "all_indices_viewpoint", [])) and self.all_indices_viewpoint[i]:
                    full_indices = list(self.all_indices_viewpoint[i])
                elif i < len(self.matching_indices):
                    full_indices = list(self.matching_indices[i])

                if len(full_indices) == 0:
                    had_active_goal = (
                        i < len(self.last_published_goal)
                        and self.last_published_goal[i] is not None
                    )
                    self.get_logger().warn(
                        f"drone {i} has no candidate viewpoint indices, freeze current position"
                    )
                    self._append_event_log(
                        "empty_candidates",
                        drone_id=int(i),
                        reason="no_candidate_indices",
                    )
                    hold_rc, _ = self._freeze_drone_goal(
                        drone_id=i,
                        reason="no_candidate_indices",
                        publish_hold=had_active_goal,
                        append_slot=True,
                    )
                    temp_delete_viewpoint.append((int(hold_rc[0]), int(hold_rc[1])))
                    continue
                elif self.astar_target.get(uav_key):
                    temp_delete_viewpoint.append(self.astar_target[uav_key])
                    view_point_x = self.astar_target[uav_key][0]
                    view_point_y = self.astar_target[uav_key][1]
                else:
                    decode_indices = list(full_indices) if full_indices else []
                    if not decode_indices and i < len(self.matching_indices) and self.matching_indices[i]:
                        decode_indices = list(self.matching_indices[i])

                    if not decode_indices:
                        had_active_goal = (
                            i < len(self.last_published_goal)
                            and self.last_published_goal[i] is not None
                        )
                        self.get_logger().warn(
                            f"drone {i} has empty viewpoint feature set, freeze current position"
                        )
                        self._append_event_log(
                            "empty_candidates",
                            drone_id=int(i),
                            reason="empty_viewpoint_features",
                        )
                        hold_rc, _ = self._freeze_drone_goal(
                            drone_id=i,
                            reason="empty_viewpoint_features",
                            publish_hold=had_active_goal,
                            append_slot=True,
                        )
                        temp_delete_viewpoint.append((int(hold_rc[0]), int(hold_rc[1])))
                        continue
                    direction = max(0, min(direction, len(decode_indices) - 1))
                    graph_vp_index = int(decode_indices[direction]) - int(self.frontier_num)
                    vp_items = self._ordered_viewpoint_items(i)
                    if not vp_items:
                        had_active_goal = (
                            i < len(self.last_published_goal)
                            and self.last_published_goal[i] is not None
                        )
                        self.get_logger().warn(
                            f"drone {i} has empty viewpoint feature set, freeze current position"
                        )
                        self._append_event_log(
                            "empty_candidates",
                            drone_id=int(i),
                            reason="empty_viewpoint_features",
                        )
                        hold_rc, _ = self._freeze_drone_goal(
                            drone_id=i,
                            reason="empty_viewpoint_features",
                            publish_hold=had_active_goal,
                            append_slot=True,
                        )
                        temp_delete_viewpoint.append((int(hold_rc[0]), int(hold_rc[1])))
                        continue

                    graph_vp_index = max(0, min(graph_vp_index, len(vp_items) - 1))
                    vp_key, value = vp_items[graph_vp_index]
                    view_point_x = int(value[0])
                    view_point_y = int(value[1])
                    original_graph_index = int(decode_indices[direction]) if direction < len(decode_indices) else None
                    original_vp = (int(view_point_x), int(view_point_y))
                    neighbor_indices = list(self.matching_indices[i]) if i < len(self.matching_indices) else []

                    banned_coords = set()
                    selected_matches_anchor = (
                        anchor_rc is not None
                        and (int(view_point_x), int(view_point_y)) == (int(anchor_rc[0]), int(anchor_rc[1]))
                    )
                    selected_matches_odom = (
                        odom_rc[0] is not None
                        and (int(view_point_x), int(view_point_y)) == (int(odom_rc[0]), int(odom_rc[1]))
                    )
                    if selected_matches_anchor and anchor_rc is not None:
                        banned_coords.add((int(anchor_rc[0]), int(anchor_rc[1])))
                    if selected_matches_odom and odom_rc[0] is not None:
                        banned_coords.add((int(odom_rc[0]), int(odom_rc[1])))

                    selection_rewritten = False
                    rewrite_reason = None
                    if banned_coords or (int(view_point_x), int(view_point_y)) in claimed_viewpoints:
                        alt_coord = self._choose_alternative_neighbor_viewpoint(
                            drone_id=i,
                            uav_key=uav_key,
                            neighbor_indices=neighbor_indices,
                            claimed_viewpoints=claimed_viewpoints,
                            banned_coords=banned_coords,
                            anchor_rc=anchor_rc,
                        )
                        if alt_coord is not None:
                            view_point_x, view_point_y = int(alt_coord[0]), int(alt_coord[1])
                            selection_rewritten = True
                            if selected_matches_anchor or selected_matches_odom:
                                rewrite_reason = "reject_self_loop_neighbor"
                            else:
                                rewrite_reason = "resolve_claimed_neighbor"

                    if (view_point_x, view_point_y) in claimed_viewpoints:
                        drone_pos = self.uav_odoms.get(uav_key, (view_point_x, view_point_y))
                        candidates = []
                        for alt_key, alt_val in vp_items:
                            alt_coord = (int(alt_val[0]), int(alt_val[1]))
                            if alt_coord in claimed_viewpoints:
                                continue
                            dist = math.hypot(alt_coord[0] - int(drone_pos[0]), alt_coord[1] - int(drone_pos[1]))
                            candidates.append((dist, alt_coord[0], alt_coord[1], alt_key))
                        if candidates:
                            candidates.sort(key=lambda item: item[0])
                            _, view_point_x, view_point_y, vp_key = candidates[0]
                            selection_rewritten = True
                            if rewrite_reason is None:
                                rewrite_reason = "resolve_claimed_global"

                    claimed_viewpoints.add((int(view_point_x), int(view_point_y)))
                    temp_delete_viewpoint.append((view_point_x, view_point_y))

                raw_vp_r, raw_vp_c = int(view_point_x), int(view_point_y)
                # Runtime must preserve training semantics here:
                # the published goal is exactly the selected viewpoint node.
                after_stretch_r, after_stretch_c = raw_vp_r, raw_vp_c
                view_point_x, view_point_y = raw_vp_r, raw_vp_c
                final_r, final_c = int(view_point_x), int(view_point_y)
                neighbor_indices = list(self.matching_indices[i]) if i < len(self.matching_indices) else []
                full_indices_dbg = list(full_indices)
                action_graph_index = int(decode_indices[direction]) if direction < len(decode_indices) else None
                selected_graph_index = self._coord_to_graph_index(i, (final_r, final_c))
                neighbor_coords = self._indices_to_viewpoint_coords(i, neighbor_indices)
                full_coords = self._indices_to_viewpoint_coords(i, full_indices_dbg)
                selected_is_neighbor = [int(final_r), int(final_c)] in neighbor_coords
                selected_matches_anchor = (
                    anchor_rc is not None
                    and [int(final_r), int(final_c)] == [int(anchor_rc[0]), int(anchor_rc[1])]
                )
                selected_matches_odom = (
                    odom_rc[0] is not None
                    and [int(final_r), int(final_c)] == [int(odom_rc[0]), int(odom_rc[1])]
                )
                exec_grid, exec_anchor_used, exec_anchor_reason = self._resolve_execution_anchor(
                    (final_r, final_c),
                    anchor_rc=anchor_rc,
                )
                exec_r, exec_c = int(exec_grid[0]), int(exec_grid[1])
                if str(exec_anchor_reason) == "center_occupied" and not bool(exec_anchor_used):
                    had_active_goal = (
                        i < len(self.last_published_goal)
                        and self.last_published_goal[i] is not None
                    )
                    self.get_logger().warn(
                        f"drone {i} selected occupied viewpoint center ({final_r},{final_c}) "
                        f"with no fallback anchor; freeze current position"
                    )
                    self._append_event_log(
                        "goal_rejected",
                        drone_id=int(i),
                        reason="center_occupied_no_fallback",
                        final_grid=[int(final_r), int(final_c)],
                        anchor_rc=anchor_rc,
                        odom_rc=None if odom_rc[0] is None else [int(odom_rc[0]), int(odom_rc[1])],
                        full_indices=[int(v) for v in full_indices_dbg[:16]],
                        neighbor_indices=[int(v) for v in neighbor_indices[:16]],
                        full_coords=full_coords,
                        neighbor_coords=neighbor_coords,
                    )
                    hold_rc, _ = self._freeze_drone_goal(
                        drone_id=i,
                        reason="center_occupied_no_fallback",
                        publish_hold=had_active_goal,
                        append_slot=True,
                    )
                    temp_delete_viewpoint.append((int(hold_rc[0]), int(hold_rc[1])))
                    continue
                ros_goal = self.pose_in_ros([exec_r, exec_c], self.resolution, self.downsample_factor)
                self.get_logger().info(
                    f"GOAL_TRACE drone={i}: action={action[0][0]}, "
                    f"odom_rc={odom_rc}, raw_vp=({raw_vp_r},{raw_vp_c}), "
                    f"after_stretch=({after_stretch_r},{after_stretch_c}), "
                    f"final_grid=({final_r},{final_c}), "
                    f"exec_grid=({exec_r},{exec_c}), "
                    f"ros_goal=({ros_goal[0]:.2f},{ros_goal[1]:.2f}), "
                    f"res={self.resolution}, ds={self.downsample_factor}, "
                    f"origin=({self.map_box_min_x:.2f},{self.map_box_min_y:.2f}), "
                    f"goal_step_cells={self.goal_step_cells}, dispatch=viewpoint_direct"
                )
                self._append_event_log(
                    "goal_trace",
                    drone_id=int(i),
                    action=int(action[0][0]),
                    odom_rc=None if odom_rc[0] is None else [int(odom_rc[0]), int(odom_rc[1])],
                    anchor_rc=anchor_rc,
                    raw_vp=[int(raw_vp_r), int(raw_vp_c)],
                    after_stretch=[int(after_stretch_r), int(after_stretch_c)],
                    final_grid=[int(final_r), int(final_c)],
                    exec_grid=[int(exec_r), int(exec_c)],
                    exec_anchor_used=bool(exec_anchor_used),
                    exec_anchor_reason=str(exec_anchor_reason),
                    full_indices=[int(v) for v in full_indices_dbg[:16]],
                    neighbor_indices=[int(v) for v in neighbor_indices[:16]],
                    full_coords=full_coords,
                    neighbor_coords=neighbor_coords,
                    action_graph_index=action_graph_index,
                    selected_graph_index=selected_graph_index,
                    selected_is_neighbor=bool(selected_is_neighbor),
                    selected_matches_anchor=bool(selected_matches_anchor),
                    selected_matches_odom=bool(selected_matches_odom),
                    selection_rewritten=bool(locals().get("selection_rewritten", False)),
                    rewrite_reason=locals().get("rewrite_reason", None),
                    original_vp=locals().get("original_vp", [int(raw_vp_r), int(raw_vp_c)]),
                    ros_goal=[float(ros_goal[0]), float(ros_goal[1])],
                    resolution=float(self.resolution),
                    downsample_factor=float(self.downsample_factor),
                    origin=[float(self.map_box_min_x), float(self.map_box_min_y)],
                    goal_step_cells=int(self.goal_step_cells),
                    dispatch_mode="viewpoint_direct",
                )
                departure_rc = None
                if i < len(self.graph_pose_committed_ds):
                    departure_rc = [
                        int(self.graph_pose_committed_ds[i][0]),
                        int(self.graph_pose_committed_ds[i][1]),
                    ]
                if departure_rc is not None and departure_rc != [int(final_r), int(final_c)]:
                    if self._mark_viewpoint_consumed(departure_rc, drone_id=i, reason="departure"):
                        temp_delete_viewpoint.append((int(departure_rc[0]), int(departure_rc[1])))
                self.uav_goal_grid_map[i][0] = int(view_point_x)
                self.uav_goal_grid_map[i][1] = int(view_point_y)
                # Predict one step ahead in pose features, but keep graph
                # topology on the last committed snapshot until arrival.
                self.graph_pose_ds[i] = [int(view_point_x), int(view_point_y)]
                self.graph_pose_initialized[i] = True
                self.active_goal_ds[i] = [int(view_point_x), int(view_point_y)]
                self.active_goal_valid[i] = True
                if int(view_point_x) == prev_goal_x and int(view_point_y) == prev_goal_y:
                    self._metric_goal_same_grid += 1
                self.goal_issue_distance_m[i] = 0.0
                if i < len(self.goal_issue_grid):
                    self.goal_issue_grid[i] = [int(final_r), int(final_c)]
                if i < len(self.goal_issue_exec_grid):
                    self.goal_issue_exec_grid[i] = [int(exec_r), int(exec_c)]
                if i < len(self.goal_issue_exec_reason):
                    self.goal_issue_exec_reason[i] = str(exec_anchor_reason)
                if i < len(self.goal_issue_odom_grid):
                    self.goal_issue_odom_grid[i] = (
                        None if odom_rc[0] is None else [int(odom_rc[0]), int(odom_rc[1])]
                    )
                if i < len(self.goal_issue_odom_xy):
                    self.goal_issue_odom_xy[i] = (
                        None
                        if uav_key not in self.uav_ros_odoms
                        else [
                            float(self.uav_ros_odoms[uav_key][0]),
                            float(self.uav_ros_odoms[uav_key][1]),
                        ]
                    )
                if i < len(self.goal_last_dist_m):
                    self.goal_last_dist_m[i] = float("inf")
                if i < len(self.goal_stall_counts):
                    self.goal_stall_counts[i] = 0
                self.arrive_flag[i] = False
                self.arrive_stable_counts[i] = 0
            else:
                view_point_x = self.uav_goal_grid_map[i][0]
                view_point_y = self.uav_goal_grid_map[i][1]
                exec_r = int(self.goal_exec_grid[i][0]) if i < len(self.goal_exec_grid) else int(view_point_x)
                exec_c = int(self.goal_exec_grid[i][1]) if i < len(self.goal_exec_grid) else int(view_point_y)
                temp_delete_viewpoint.append((view_point_x, view_point_y))

            if refresh:
                self.goal_exec_grid[i] = [int(exec_r), int(exec_c)]
            self.goal_of_drone_msg.append(self.To_goal_msgs(self.pose_in_ros([exec_r, exec_c], self.resolution, self.downsample_factor)))
            self.goal_of_drone.append(self.pose_in_ros([exec_r, exec_c], self.resolution, self.downsample_factor))
            self.goal_of_drone_grid.append([view_point_x, view_point_y])
        return temp_delete_viewpoint

    def each_uav_arrive(self):
        for i, uav_id in enumerate(self.uav_odoms.keys()):
            if i >= len(self.goal_of_drone_msg) or self.goal_of_drone_msg[i] is False:
                self.arrive_flag[i] = True
                self.arrive_stable_counts[i] = 0
                continue
            if i >= len(self.goal_of_drone):
                continue
            if uav_id not in self.uav_ros_odoms:
                self.arrive_flag[i] = False
                self.arrive_stable_counts[i] = 0
                continue
            goal_grid = None
            odom_grid = None
            if i < len(self.goal_of_drone_grid):
                goal_grid = (
                    int(self.goal_of_drone_grid[i][0]),
                    int(self.goal_of_drone_grid[i][1]),
                )
            exec_grid = None
            if i < len(self.goal_exec_grid):
                exec_grid = (
                    int(self.goal_exec_grid[i][0]),
                    int(self.goal_exec_grid[i][1]),
                )
            if uav_id in self.uav_odoms:
                odom_grid = (
                    int(self.uav_odoms[uav_id][0]),
                    int(self.uav_odoms[uav_id][1]),
                )
            # Align runtime with training semantics: once the robot enters the
            # selected viewpoint cell, the next graph rebuild should happen
            # from that cell rather than waiting for precise world-frame center
            # convergence and low speed.
            if goal_grid is not None and odom_grid is not None and (
                odom_grid == goal_grid or (exec_grid is not None and odom_grid == exec_grid)
            ):
                if not self.arrive_flag[i]:
                    self._append_event_log(
                        "arrival_cell_match",
                        drone_id=int(i),
                        goal_grid=[int(goal_grid[0]), int(goal_grid[1])],
                        odom_grid=[int(odom_grid[0]), int(odom_grid[1])],
                        exec_grid=None if exec_grid is None else [int(exec_grid[0]), int(exec_grid[1])],
                    )
                self.arrive_stable_counts[i] = self.arrival_stable_cycles
                self.arrive_flag[i] = True
                continue
            if goal_grid is not None and not getattr(self, "manual_step_mode", False):
                segment_state = self._segment_completion_state(i, uav_id, goal_grid)
                if (
                    segment_state is not None
                    and float(segment_state["progress_ratio"]) >= float(self.arrival_segment_progress_ratio)
                    and float(segment_state["cross_track_m"]) <= float(self.arrival_cross_track_tol_m)
                ):
                    if not self.arrive_flag[i]:
                        self._append_event_log(
                            "arrival_segment_complete",
                            drone_id=int(i),
                            start_grid=list(segment_state["start_grid"]),
                            goal_grid=list(segment_state["goal_grid"]),
                            odom_grid=[int(odom_grid[0]), int(odom_grid[1])] if odom_grid is not None else None,
                            progress_ratio=float(segment_state["progress_ratio"]),
                            cross_track_m=float(segment_state["cross_track_m"]),
                        )
                    self.arrive_stable_counts[i] = self.arrival_stable_cycles
                    self.arrive_flag[i] = True
                    continue
            dx = self.goal_of_drone[i][0] - self.uav_ros_odoms[uav_id][0]
            dy = self.goal_of_drone[i][1] - self.uav_ros_odoms[uav_id][1]
            dist_l2 = float(np.hypot(dx, dy))
            speed = float(self.uav_speeds.get(uav_id, 0.0))
            issued_dist = self.goal_issue_distance_m[i] if i < len(self.goal_issue_distance_m) else 0.0
            pos_tol = adaptive_arrival_tolerance(
                issued_dist_m=issued_dist,
                map_based_tol_m=self.arrive_distance_threshold,
                max_pos_tol_m=self.arrival_pos_tol_m,
                min_pos_tol_m=self.arrival_min_pos_tol_m,
                adaptive_ratio=self.arrival_adaptive_ratio,
            )
            close_enough = dist_l2 <= pos_tol
            slow_enough = speed <= self.arrival_vel_tol_mps

            if close_enough and slow_enough:
                self.arrive_stable_counts[i] += 1
            else:
                self.arrive_stable_counts[i] = 0

            self.arrive_flag[i] = self.arrive_stable_counts[i] >= self.arrival_stable_cycles

        return self.arrive_flag
