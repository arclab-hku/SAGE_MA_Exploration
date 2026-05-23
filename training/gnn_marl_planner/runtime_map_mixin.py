from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from colorama import Fore, Style
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid
from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
from skimage.measure import block_reduce
from .uav_index import uav_key_from_index

class RuntimeMapMixin:
    def _ordered_viewpoint_items(self, drone_id: int):
        if drone_id < 0 or drone_id >= len(getattr(self, "viewpoint_node_feature", [])):
            return []
        vp_dict = self.viewpoint_node_feature[drone_id]
        if not vp_dict:
            return []
        ordered_keys = sorted(vp_dict.keys())
        return [(key, vp_dict[key]) for key in ordered_keys]

    def _indices_batch_to_coords(self, indices_batch, limit: int = 16):
        coords_batch = []
        for drone_id, indices in enumerate(indices_batch):
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
            coords_batch.append(coords)
        return coords_batch

    def _append_event_log(self, event: str, **fields):
        log_path = getattr(self, "event_log_path", None)
        if not log_path:
            return
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ts": time.time(), "event": event}
            payload.update(fields)
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def _graph_age_info(self):
        now = time.monotonic()
        def _age(ts):
            if not ts:
                return None
            return float(max(0.0, now - ts))
        return {
            "layer2_age_s": _age(getattr(self, "_last_layer2_delta_ts", 0.0)),
            "graph_age_s": _age(getattr(self, "_last_graph_delta_ts", 0.0)),
            "snapshot_age_s": _age(getattr(self, "_last_snapshot_commit_ts", 0.0)),
        }

    def _graph_map_evidence_info(self):
        info = {
            "free_cells": 0,
            "known_cells": 0,
            "unknown_cells": 0,
            "min_local_free": None,
        }
        data = getattr(self, "downsampled_map_data", None)
        if data is None or getattr(data, "size", 0) == 0:
            return info

        info["free_cells"] = int(np.count_nonzero(data == 0))
        info["known_cells"] = int(np.count_nonzero(data != -1))
        info["unknown_cells"] = int(np.count_nonzero(data == -1))

        radius = int(getattr(self, "startup_local_free_radius_cells", 0))
        local_counts = []
        for uav_key, odom in getattr(self, "uav_odoms", {}).items():
            if odom is None or len(odom) < 2:
                continue
            r = int(odom[0])
            c = int(odom[1])
            r0 = max(0, r - radius)
            r1 = min(data.shape[0], r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(data.shape[1], c + radius + 1)
            local_counts.append(int(np.count_nonzero(data[r0:r1, c0:c1] == 0)))
        if local_counts:
            info["min_local_free"] = int(min(local_counts))
        return info

    def _graph_startup_gate_info(self):
        counts = self._graph_node_counts() if hasattr(self, "_graph_node_counts") else {}
        evidence = self._graph_map_evidence_info()
        topo_updates = int(getattr(self, "_graph_topology_updates", 0))
        frontiers = int(counts.get("frontier", 0))
        viewpoints = int(counts.get("viewpoint", 0))
        free_cells = int(evidence.get("free_cells", 0))
        min_local_free = evidence.get("min_local_free")

        need_topo_updates = int(getattr(self, "startup_min_graph_updates", 0))
        need_frontiers = int(getattr(self, "startup_min_frontiers", 0))
        need_viewpoints = int(getattr(self, "startup_min_viewpoints", 0))
        configured_need_free_cells = int(getattr(self, "startup_min_free_cells", 0))
        need_local_free = int(getattr(self, "startup_min_local_free_cells", 0))

        if getattr(self, "map_input_mode", "") == "graph_sync":
            need_free_cells = 0
            enough_free = True
        else:
            need_free_cells = configured_need_free_cells
            enough_free = free_cells >= need_free_cells

        enough_updates = topo_updates >= need_topo_updates
        enough_frontiers = frontiers >= need_frontiers
        enough_viewpoints = viewpoints >= need_viewpoints
        if need_local_free <= 0:
            enough_local_free = True
        else:
            enough_local_free = (
                min_local_free is not None and int(min_local_free) >= need_local_free
            )

        ready = (
            enough_updates
            and enough_frontiers
            and enough_viewpoints
            and enough_free
            and enough_local_free
        )
        return {
            "ready": bool(ready),
            "topo_updates": topo_updates,
            "frontiers": frontiers,
            "viewpoints": viewpoints,
            "free_cells": free_cells,
            "min_local_free": min_local_free,
            "need_topo_updates": need_topo_updates,
            "need_frontiers": need_frontiers,
            "need_viewpoints": need_viewpoints,
            "need_free_cells": need_free_cells,
            "configured_need_free_cells": configured_need_free_cells,
            "need_local_free": need_local_free,
        }

    def publish_occupancy_map(self, rgb_map):
        msg = UInt8MultiArray()
        msg.data = rgb_map.flatten().tolist()  # 将 3D 数组展平为 1D 列表
        
        # 设置数组的维度信息
        msg.layout.dim = [
            MultiArrayDimension(label="height", size=rgb_map.shape[0], stride=rgb_map.shape[0]*rgb_map.shape[1]*3),
            MultiArrayDimension(label="width", size=rgb_map.shape[1], stride=rgb_map.shape[1]*3),
            MultiArrayDimension(label="channel", size=3, stride=3)
        ]
        
        self.publisher.publish(msg)
        self.get_logger().debug("RGB occupancy map published")

    def maybe_publish_rgb(self):
        if not self.publish_rgb_map:
            return
        now = time.monotonic()
        if (now - self._last_rgb_publish_ts) < self.rgb_publish_period_s:
            return
        self.occupancy_to_rgb(self.downsampled_map_data)
        self._last_rgb_publish_ts = now

    def has_active_goals(self) -> bool:
        return any(msg is not False for msg in self.goal_of_drone_msg)

    def maybe_log_status(self):
        now = time.monotonic()
        self._status_loop_counter += 1
        elapsed = now - self._last_status_log_ts
        if elapsed < self.status_log_period_s:
            return
        loop_hz = self._status_loop_counter / max(elapsed, 1e-6)
        self.get_logger().info(
            "status: "
            f"loop_hz={loop_hz:.1f}, map_update={self.map_update_flag}, "
            f"frontier={len(self.frontier_dict)}, viewpoint={len(self.viewpoint_dict)}, "
            f"active_goals={sum(1 for m in self.goal_of_drone_msg if m is not False)}, "
            f"manual_step={getattr(self, 'manual_step_mode', False)}, "
            f"step_pending={getattr(self, 'manual_step_pending', False)}, "
            f"plan_ms={self._last_profile['total_ms']:.1f}, "
            f"map_ms={self._last_profile['map_ms']:.1f}, "
            f"rebuild_ms={self._last_profile['rebuild_ms']:.1f}, "
            f"feature_ms={self._last_profile['feature_ms']:.1f}, "
            f"infer_ms={self._last_profile['infer_ms']:.1f}, "
            f"goal_ms={self._last_profile['goal_ms']:.1f}, "
            f"replan_ms={self._last_replan_compute_ms:.1f}, "
            f"exit={self._last_profile['exit']}, "
            f"replan(total/succ/empty)={self._metric_replan_total}/{self._metric_replan_success}/{self._metric_replan_empty}, "
            f"goal_pub(total/dup_skip/same_grid)={self._metric_goal_publish_total}/{self._metric_goal_publish_dup_skip}/{self._metric_goal_same_grid}, "
            f"goal_arrival_ema_s={self._goal_arrival_ema_s:.2f}"
        )
        if self.map_input_mode == "graph_sync":
            ages = self._graph_age_info()
            counts = self._graph_node_counts() if hasattr(self, "_graph_node_counts") else {}
            evidence = self._graph_map_evidence_info()
            self.get_logger().info(
                "graph_status: "
                f"ready={getattr(self, '_startup_graph_ready', False)}, "
                f"topo_updates={getattr(self, '_graph_topology_updates', 0)}, "
                f"snapshot_committed={getattr(self, '_graph_snapshot_committed_once', False)}, "
                f"nodes={counts.get('nodes', 0)}, frontiers={counts.get('frontier', 0)}, viewpoints={counts.get('viewpoint', 0)}, "
                f"free_cells={evidence['free_cells']}, known_cells={evidence['known_cells']}, "
                f"min_local_free={evidence['min_local_free'] if evidence['min_local_free'] is not None else -1}, "
                f"layer2_age_s={ages['layer2_age_s'] if ages['layer2_age_s'] is not None else -1.0:.3f}, "
                f"graph_age_s={ages['graph_age_s'] if ages['graph_age_s'] is not None else -1.0:.3f}, "
                f"snapshot_age_s={ages['snapshot_age_s'] if ages['snapshot_age_s'] is not None else -1.0:.3f}"
            )
        self._status_loop_counter = 0
        self._last_status_log_ts = now

    def _set_profile(
        self,
        total_ms: float,
        map_ms: float,
        rebuild_ms: float,
        feature_ms: float,
        infer_ms: float,
        goal_ms: float,
        exit_reason: str,
    ) -> None:
        self._last_profile = {
            "total_ms": float(total_ms),
            "map_ms": float(map_ms),
            "rebuild_ms": float(rebuild_ms),
            "feature_ms": float(feature_ms),
            "infer_ms": float(infer_ms),
            "goal_ms": float(goal_ms),
            "frontier": len(self.frontier_dict),
            "viewpoint": len(self.viewpoint_dict),
            "exit": str(exit_reason),
        }

    def _init_graph_sync_map(self):
        """Initialize Layer2 state arrays when map input comes from graph_sync."""
        cell_size = float(self.resolution) * float(self.downsample_factor)
        width_m = float(self.map_box_max_x) - float(self.map_box_min_x)
        height_m = float(self.map_box_max_y) - float(self.map_box_min_y)
        self.map_size_width = max(2, int(round(width_m / cell_size)))
        self.map_size_height = max(2, int(round(height_m / cell_size)))
        self._allocate_graph_sync_arrays(cell_size)

    def _allocate_graph_sync_arrays(self, cell_size: float):
        self.width = self.map_size_width
        self.height = self.map_size_height

        self.downsampled_map_data = -1 * np.ones(
            (self.map_size_height, self.map_size_width), dtype=np.int8
        )
        self.downsampled_map_data[0, :] = 100
        self.downsampled_map_data[-1, :] = 100
        self.downsampled_map_data[:, 0] = 100
        self.downsampled_map_data[:, -1] = 100

        self._layer2_obs_count = np.zeros(
            (self.map_size_height, self.map_size_width), dtype=np.uint8
        )
        self._layer2_ver_lamport = np.zeros(
            (self.map_size_height, self.map_size_width), dtype=np.uint32
        )
        self._layer2_ver_source = np.full(
            (self.map_size_height, self.map_size_width), -1, dtype=np.int32
        )
        self.get_logger().info(
            f"graph_sync map initialized: h={self.map_size_height}, w={self.map_size_width}, cell={cell_size:.3f}m"
        )

    def _apply_graph_sync_geometry(self, width: int, height: int, res: float, ox: float, oy: float, source: str):
        changed = (
            (not self._layer2_meta_applied)
            or width != int(self.map_size_width)
            or height != int(self.map_size_height)
            or abs(res - float(self.resolution)) > 1e-6
            or abs(ox - float(self.map_box_min_x)) > 1e-6
            or abs(oy - float(self.map_box_min_y)) > 1e-6
        )
        self._layer2_meta_applied = True

        if not changed:
            return

        # GraphSync geometry is expressed directly in the downsampled grid.
        self.map_box_min_x = ox
        self.map_box_min_y = oy
        self.resolution = res
        self.downsample_factor = 1.0
        self.map_box_max_x = ox + width * res
        self.map_box_max_y = oy + height * res
        self.arrive_distance_threshold = 0.9 * self.resolution * self.downsample_factor
        self.viewrange = max(6, int(self.max_ray_length / (self.resolution * self.downsample_factor)))
        self.goal_step_cells = max(1, int(self.goal_step_m / (self.resolution * self.downsample_factor)))
        self.min_goal_delta_cells = max(1, int(self.min_goal_delta_m / (self.resolution * self.downsample_factor)))
        self.map_size_width = width
        self.map_size_height = height
        self._allocate_graph_sync_arrays(cell_size=self.resolution * self.downsample_factor)
        self.map_update_flag = True
        self.get_logger().info(
            f"{source} applied: w={width}, h={height}, res={res:.3f}, "
            f"origin=({ox:.3f},{oy:.3f}), viewrange={self.viewrange} cells, "
            f"goal_step={self.goal_step_cells} cells, min_goal_delta={self.min_goal_delta_cells} cells"
        )

    def _on_layer2_meta(self, msg: MapMetaData):
        if self.map_input_mode != "graph_sync":
            return
        try:
            width = int(msg.width)
            height = int(msg.height)
            res = float(msg.resolution)
            ox = float(msg.origin.position.x)
            oy = float(msg.origin.position.y)
            if width <= 0 or height <= 0 or res <= 0.0:
                return

            self._layer2_meta_recv_count += 1
            self._apply_graph_sync_geometry(width, height, res, ox, oy, "layer2_meta")
        except Exception as e:
            self.get_logger().error(f"failed to apply layer2_meta: {e}")

    def _on_graph_sync_shadow_map(self, msg: OccupancyGrid):
        if self.map_input_mode != "graph_sync" or getattr(self, "_layer2_meta_applied", False):
            return
        try:
            raw_w = int(msg.info.width)
            raw_h = int(msg.info.height)
            raw_res = float(msg.info.resolution)
            ox = float(msg.info.origin.position.x)
            oy = float(msg.info.origin.position.y)
            if raw_w <= 0 or raw_h <= 0 or raw_res <= 0.0:
                return

            ds = max(1, int(round(float(getattr(self, "graph_sync_launch_downsample", 1.0)))))
            width = int(math.ceil(raw_w / float(ds)))
            height = int(math.ceil(raw_h / float(ds)))
            res = raw_res * float(ds)
            self._apply_graph_sync_geometry(width, height, res, ox, oy, "shadow_map_fallback")
        except Exception as e:
            self.get_logger().error(f"failed to apply graph_sync shadow map fallback: {e}")

    def _state_priority(state: int) -> int:
        if state == 100:
            return 2
        if state == 0:
            return 1
        return 0

    def _graph_node_priority(node_type: int) -> int:
        if node_type == 0:  # frontier
            return 2
        if node_type == 1:  # viewpoint
            return 1
        return 0

    def _reset_frontier_bounds(self):
        self.min_frontier_x = float('inf')
        self.max_frontier_x = 0
        self.min_frontier_y = float('inf')
        self.max_frontier_y = 0

    def listener_callback0(self, msg):
        self.width = msg.info.width
        self.height = msg.info.height
        # self.resolution = msg.info.resolution
        self.origin_map_data0 = np.array(msg.data).reshape((self.height, self.width))
        self.get_logger().info(f'{Fore.GREEN}Map height and width are {self.height}, {self.width} {Style.RESET_ALL}')
        # 计算下采样后的宽度和高度
        self.map_size_width = new_width = int(self.width / self.downsample_factor)
        self.map_size_height = new_height = int(self.height / self.downsample_factor)
        # 更新地图
        self.map_data['0'] = block_reduce(self.origin_map_data0, block_size=(self.downsample_factor, self.downsample_factor), func=np.max)

        # 使用 scipy.ndimage.zoom 进行下采样
        # self.map_data['0'] = np.zeros((new_height, new_width), dtype=np.int8)
        # for i in range(new_height):
        #     for j in range(new_width):
        #         start_x = i * self.downsample_factor
        #         end_x = (i + 1) * self.downsample_factor
        #         start_y = j * self.downsample_factor
        #         end_y = (j + 1) * self.downsample_factor
        #         self.map_data['0'][i, j] = np.max(self.origin_map_data0[start_x:end_x, start_y:end_y])
        # 更新宽度和高度
        self.width = new_width
        self.height = new_height
        self.get_logger().info(f'{Fore.GREEN}self.initiate_map_flag: {self.initiate_map_flag}{Style.RESET_ALL}')
        try: 
            if self.initiate_map_flag[0]:
                self.initiate_map_flag[0] = False
                self.get_logger().info(f'self.map_size_height is {self.map_size_height}, self.map_size_width is {self.map_size_width}')
                self.downsampled_map_data = -1 * np.ones((self.map_size_height, self.map_size_width), dtype=np.int8)
                # 边缘为100， 设置为障碍物
                self.downsampled_map_data[0, :] = 100
                self.downsampled_map_data[-1, :] = 100
                self.downsampled_map_data[:, 0] = 100
                self.downsampled_map_data[:, -1] = 100
                
                # self.get_logger().info(f'{Fore.GREEN}self.downsampled_map_data is done{Style.RESET_ALL}')
                self.map_update_flag = True # 宣布地图数据已经准备好了，可以开始根基viewrange更新下采样地图了
        except Exception as e:
            # 打印具体错误信息，比如什么str(e)
            self.get_logger().error(f'{Fore.RED}Error in listener_callback0: {str(e)}{Style.RESET_ALL}')

    def listener_callback1(self, msg):
        self.width = msg.info.width
        self.height = msg.info.height
        # self.resolution = msg.info.resolution
        self.origin_map_data1 = np.array(msg.data).reshape((self.height, self.width))
        self.get_logger().info(f'{Fore.GREEN}Map height and width are {self.height}, {self.width} {Style.RESET_ALL}')
        # 计算下采样后的宽度和高度
        # time.sleep(100000)
        new_width = int(self.width / self.downsample_factor)
        new_height = int(self.height / self.downsample_factor)
        self.map_data['1'] = block_reduce(self.origin_map_data1, block_size=(self.downsample_factor, self.downsample_factor), func=np.max)
        # 使用 scipy.ndimage.zoom 进行下采样
        # self.map_data['1'] = np.zeros((new_height, new_width), dtype=np.int8)
        # for i in range(new_height):
        #     for j in range(new_width):
        #         start_x = i * self.downsample_factor
        #         end_x = (i + 1) * self.downsample_factor
        #         start_y = j * self.downsample_factor
        #         end_y = (j + 1) * self.downsample_factor
        #         self.map_data['1'][i, j] = np.max(self.origin_map_data1[start_x:end_x, start_y:end_y])
        try: 
            if self.initiate_map_flag[1]:
                self.initiate_map_flag[1] = False
        except Exception as e:
            # 打印具体错误信息，比如什么str(e)
            self.get_logger().error(f'{Fore.RED}Error in listener_callback1: {str(e)}{Style.RESET_ALL}')

    def get_uav_odom(self, msg, uav_id):
        ros_x = msg.pose.pose.position.x 
        ros_y = msg.pose.pose.position.y
        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        speed = float(np.hypot(vel_x, vel_y))
        self.get_logger().debug(f'uav_id: {uav_id}, ros_x: {ros_x}, ros_y: {ros_y}')
        map_col = int((ros_x - self.map_box_min_x) / (self.resolution * self.downsample_factor))
        map_row = int((ros_y - self.map_box_min_y) / (self.resolution * self.downsample_factor))
        self.uav_ros_odoms[uav_id] = (ros_x, ros_y)
        self.uav_speeds[uav_id] = speed
        idx = None
        try:
            raw_idx = int(uav_id)
            if 0 <= raw_idx < len(self.first_update_map_flag):
                idx = raw_idx
            elif 1 <= raw_idx <= len(self.first_update_map_flag):
                # Compatibility for 1-based topic suffixes (quad_1, quad_2, ...)
                idx = raw_idx - 1
        except Exception:
            idx = None
        if self.map_input_mode == "graph_sync":
            self.uav_odoms[uav_id] = (map_row, map_col)
            if uav_id not in self.astar_target:
                self.astar_target[uav_id] = None
            if idx is not None and 0 <= idx < len(self.uav_goal_grid_map):
                if not self.goal_grid_initialized[idx]:
                    self.uav_goal_grid_map[idx][0] = int(map_row)
                    self.uav_goal_grid_map[idx][1] = int(map_col)
                    self.goal_grid_initialized[idx] = True
            if idx is not None:
                self.first_update_map_flag[idx] = False
            return
        if idx is None:
            return
        if self.first_update_map_flag[idx]:
            self.uav_odoms[uav_id] = (map_row, map_col)
            self.astar_target[uav_id] = None
            if not self.goal_grid_initialized[idx]:
                self.uav_goal_grid_map[idx][0] = int(map_row)
                self.uav_goal_grid_map[idx][1] = int(map_col)
                self.goal_grid_initialized[idx] = True
            if not self.initiate_map_flag[idx]: 
                self.update_downsampled_map(map_row, map_col, uav_id)
                self.first_update_map_flag[idx] = False

    def update_downsampled_map(self, map_row, map_col, uav_id):
        if self.map_input_mode == "graph_sync":
            return
        self.get_logger().info(f'{Fore.GREEN}Start updating{Style.RESET_ALL}')
        if not self.map_update_flag:
            self.get_logger().info(f'{Fore.RED}self.map_update_flag is False{Style.RESET_ALL}')
            return
        if len(self.map_data) < self.drone_num:
            self.get_logger().info(f'{Fore.RED}self.map_data length is less than drone num{Style.RESET_ALL}')
            return
    
        self.get_logger().info(f'{Fore.GREEN}map row and map col are{map_row} {map_col}{Style.RESET_ALL}')
        height_0 = max(0, map_row - self.viewrange)
        height_1 = min(self.map_size_height, map_row + self.viewrange + 1)
        width_0 = max(0, map_col - self.viewrange)
        width_1 = min(self.map_size_width, map_col + self.viewrange + 1)

        # 创建局部坐标网格 (仅在 [height_0:height_1, width_0:width_1] 范围)
        height_idx, width_idx = np.ogrid[height_0:height_1, width_0:width_1]
        dist_from_center = np.sqrt((height_idx - map_row)**2 + (width_idx - map_col)**2)
        
        # 距离掩码 - 在视野范围内的点
        distance_mask = dist_from_center < self.viewrange
        # 排除已知障碍物
        initial_mask = np.logical_and(distance_mask, self.downsampled_map_data[height_0:height_1, width_0:width_1] != 100)
        
        # 由于ray_point函数本身难以矢量化，我们可以使用更简单的光线投射近似
        # 这种方法基于：如果从中心到某点的直线上有障碍物，那么该点不可见
        
        # 获取所有在距离范围内的点的坐标
        points_y, points_x = np.where(initial_mask)
        points_y += height_0  # 调整回全局坐标
        points_x += width_0   # 调整回全局坐标
        
        # 创建可见性掩码数组
        visibility_mask = np.zeros_like(initial_mask, dtype=bool)
        
        if len(points_y) > 0:
            # 创建从中心点到每个点的单位向量数组
            dy = points_y - map_row
            dx = points_x - map_col
            
            # 计算距离
            distances = np.sqrt(dy**2 + dx**2)
            
            # 防止除以零
            non_zero_distances = np.maximum(distances, 1e-10)
            
            # 单位向量 
            unit_y = dy / non_zero_distances
            unit_x = dx / non_zero_distances
            
            # 对每个点，我们只检查障碍物点
            # 创建一个包含所有障碍物点的掩码
            obstacle_mask = self.downsampled_map_data == 100
            obstacle_y, obstacle_x = np.where(obstacle_mask)
            
            # 计算每个点是否可见
            for i in range(len(points_y)):
                pt_y, pt_x = points_y[i], points_x[i]
                u_y, u_x = unit_y[i], unit_x[i]
                dist = distances[i]
                
                # 找出可能阻挡这条光线的障碍物
                # 首先，优化性能，我们只考虑那些在当前点与中心点构成的矩形范围内的障碍物
                min_y = min(map_row, pt_y)
                max_y = max(map_row, pt_y)
                min_x = min(map_col, pt_x)
                max_x = max(map_col, pt_x)
                
                # 找出在这个矩形范围内的障碍物
                mask_in_range = (obstacle_y >= min_y) & (obstacle_y <= max_y) & (obstacle_x >= min_x) & (obstacle_x <= max_x)
                obstacles_in_range_y = obstacle_y[mask_in_range]
                obstacles_in_range_x = obstacle_x[mask_in_range]
                
                # 如果没有障碍物在范围内，则点可见
                if len(obstacles_in_range_y) == 0:
                    local_y = pt_y - height_0
                    local_x = pt_x - width_0
                    visibility_mask[local_y, local_x] = True
                    continue
                
                # 计算障碍物到中心点的向量
                obs_dy = obstacles_in_range_y - map_row
                obs_dx = obstacles_in_range_x - map_col
                
                # 计算障碍物到中心点的距离
                obs_dist = np.sqrt(obs_dy**2 + obs_dx**2)
                
                # 计算障碍物点到光线的距离
                # 使用向量叉积计算点到线的距离
                cross_product = np.abs(obs_dx * u_y - obs_dy * u_x)
                point_line_dist = cross_product
                
                # 设置一个阈值，判断障碍物是否在光线上
                # 实际应用时，可能需要根据分辨率调整这个阈值
                threshold = 0.5
                
                # 找出距离光线够近且距离中心点小于当前点的障碍物
                blocking_obstacles = (point_line_dist < threshold) & (obs_dist < dist)
                
                # 如果没有阻挡的障碍物，则点可见
                if not np.any(blocking_obstacles):
                    local_y = pt_y - height_0
                    local_x = pt_x - width_0
                    visibility_mask[local_y, local_x] = True
        
        # 符合这个条件的区域不更新，防止由于某些原因出错: 如果downsampled_map_data == 0 但是 self.map_data == -1
        not_update_area = np.logical_and(self.downsampled_map_data[height_0:height_1, width_0:width_1] == 0, 
                                         self.map_data[uav_id][height_0:height_1, width_0:width_1] == -1)
        
        # 最终更新区域：在视野范围内 + 光线未被阻挡 + 不在不更新区域
        update_area = visibility_mask & ~not_update_area
        
        # 把 self.map_data 对应区域的值先复制到 downsampled_map_data
        self.downsampled_map_data[height_0:height_1, width_0:width_1][update_area] = \
            self.map_data[uav_id][height_0:height_1, width_0:width_1][update_area]
        
        # 障碍物总是被更新
        obstacle_mask = np.where(self.map_data[uav_id][height_0:height_1, width_0:width_1] == 100) 
        self.downsampled_map_data[height_0:height_1, width_0:width_1][obstacle_mask] = 100    

        # 在这里判断一下self.map_data[height_0:height_1, width_0:width_1][mask]中有多少-1， 0， 100
        self.get_logger().info(f"self.map_data[{uav_id}][height_0:height_1, width_0:width_1][mask]中有多少-1， 0， 100: {np.unique(self.map_data[uav_id][height_0:height_1, width_0:width_1][update_area], return_counts=True)}")
        self.get_logger().info(f"self.downsampled_map_data[height_0:height_1, width_0:width_1][mask]中有多少-1， 0， 100: {np.unique(self.downsampled_map_data[height_0:height_1, width_0:width_1][update_area], return_counts=True)}")
        # 日志输出
        self.get_logger().info(f"更新完成: 下采样地图局部范围=({height_0}:{height_1}, {width_0}:{width_1})")

    def occupancy_to_rgb(self, data):
        rgb_map = np.zeros((self.map_size_height, self.map_size_width, 3), dtype=np.uint8)
        
        # 设置未知区域为灰色 (128, 128, 128)
        rgb_map[data == -1] = [128, 128, 128]
        
        # 设置已知区域
        known = data != -1
        # 将0-100的范围映射到0-255
        values = np.interp(data[known], [0, 100], [255, 0]).astype(np.uint8)
        rgb_map[known] = np.stack([values, values, values], axis=-1)
        # self.get_logger().info(f'frontier_dict: {self.frontier_dict}')
        # self.get_logger().info(f'viewpoint_dict: {self.viewpoint_dict}')
        for frontier in self.frontier_dict:
            rgb_map[frontier[0], frontier[1]] = [0, 255, 0]
        
        for view_point in self.viewpoint_dict:
            rgb_map[view_point[0], view_point[1]] = [0, 0, 255]

        if  self.goal_of_drone:
            for goal in self.goal_of_drone_grid:
                rgb_map[int(goal[0]), int(goal[1])] = [128, 0, 128]
        try:
            for uav_id in self.uav_odoms:
                # self.get_logger().info(f'self.uav_odoms[uav_id]: {self.uav_odoms[uav_id]}')
                rgb_map[self.uav_odoms[uav_id][0], self.uav_odoms[uav_id][1]] = [255, 0, 0]
        except:
            self.get_logger().debug('No uav_odoms for RGB overlay')
        # self.get_logger().info(f"{Fore.GREEN}frontier_dict: {self.frontier_dict}{Style.RESET_ALL}")
        # 在原图上绘制红色圆圈
        # rgb_map[bottom:top, left:right][circle_mask] = [255, 0, 0]
        
        self.publish_occupancy_map(rgb_map)

    def To_goal_msgs(self, goal):
        msg = PoseStamped()
        msg.header.frame_id = 'world'
        msg.pose.position.x = goal[0] # ros的x是列
        msg.pose.position.y = goal[1] # ros的y*是行
        msg.pose.position.z = 1.0
        msg.pose.orientation.w = 1.0
        return msg

    def pose_in_ros(self, pos, resolution, downsample_factor):
        
        # map_col * (self.resolution * self.downsample_factor)) + self.map_box_min_x = int((ros_x) 
        # self.map_box_max_y)  - map_row * (self.resolution * self.downsample_factor)) = int((-ros_y + 
        # self.uav_ros_odoms[uav_id] = (ros_x, ros_y)
        # if self.first_update_map_flag:
        #     self.uav_odoms[uav_id] = (map_row, map_col)
        
        
        
        cell_size = resolution * downsample_factor
        ros_y = (pos[0] + 0.5) * cell_size + self.map_box_min_y  # row -> world y, cell center
        ros_x = (pos[1] + 0.5) * cell_size + self.map_box_min_x  # col -> world x, cell center

        return [ros_x, ros_y]

    def _in_map(self, r: int, c: int) -> bool:
        return 0 <= r < self.map_size_height and 0 <= c < self.map_size_width

    def _effective_pos_for_graph(self, drone_idx: int, uav_key: str):
        """Return the predictive discrete pose used for graph features.

        The committed graph topology stays one step behind real sensing, while
        pose features are allowed to lead by one committed goal.  This keeps a
        stable snapshot for topology while avoiding in-between-viewpoint
        decisions.
        """
        if (
            drone_idx < len(self.graph_pose_initialized)
            and self.graph_pose_initialized[drone_idx]
        ):
            return (
                int(self.graph_pose_ds[drone_idx][0]),
                int(self.graph_pose_ds[drone_idx][1]),
            )
        if drone_idx < len(self.graph_pose_committed_ds):
            return (
                int(self.graph_pose_committed_ds[drone_idx][0]),
                int(self.graph_pose_committed_ds[drone_idx][1]),
            )
        # Fallback to real odom before first goal is dispatched
        if uav_key in self.uav_odoms:
            return (int(self.uav_odoms[uav_key][0]), int(self.uav_odoms[uav_key][1]))
        return (0, 0)

    def _stretch_goal_in_free_space(self, uav_key: str, target_r: int, target_c: int):
        """Stretch short 1-cell goals to farther free cells along the selected direction."""
        if uav_key not in self.uav_odoms:
            return target_r, target_c
        cur_r, cur_c = self.uav_odoms[uav_key]
        dr = int(np.sign(target_r - cur_r))
        dc = int(np.sign(target_c - cur_c))
        if dr == 0 and dc == 0:
            return target_r, target_c

        best_r, best_c = target_r, target_c
        for k in range(1, self.goal_step_cells + 1):
            rr = cur_r + dr * k
            cc = cur_c + dc * k
            if not self._in_map(rr, cc):
                break
            cell = int(self.downsampled_map_data[rr, cc])
            if cell == 100:
                break
            if cell == 0:
                best_r, best_c = rr, cc

        return best_r, best_c

    def update_viewrange(self):  
        has_committed_graph = (
            getattr(self, "_graph_snapshot_committed_once", False)
            or bool(getattr(self, "_graph_nodes_committed", {}))
            or bool(getattr(self, "_graph_nodes_merged", {}))
        )
        no_graph_snapshot = self.map_input_mode == "graph_sync" and not has_committed_graph
        waiting_for_map_update = self.map_input_mode != "graph_sync" and not self.map_update_flag
        if waiting_for_map_update or no_graph_snapshot or len(self.uav_odoms) < self.drone_num:
            now = time.monotonic()
            if (now - self._last_no_data_log_ts) > 2.0:
                if no_graph_snapshot:
                    self.get_logger().warn(
                        f"No committed graph snapshot yet or not enough odom: "
                        f"live_nodes={len(getattr(self, '_graph_nodes_merged', {}))}, "
                        f"uav_odoms={len(self.uav_odoms)}/{self.drone_num}"
                    )
                else:
                    self.get_logger().warn(
                        f"No map update yet or not enough odom: map_update_flag={self.map_update_flag}, "
                        f"uav_odoms={len(self.uav_odoms)}/{self.drone_num}"
                    )
                self._last_no_data_log_ts = now
            self._set_profile(
                total_ms=0.0,
                map_ms=0.0,
                rebuild_ms=0.0,
                feature_ms=0.0,
                infer_ms=0.0,
                goal_ms=0.0,
                exit_reason="no_data",
            )
            return
        if self.map_input_mode == "graph_sync" and not getattr(self, "_layer2_meta_applied", False):
            now = time.monotonic()
            if (now - self._last_no_data_log_ts) > 2.0:
                self.get_logger().warn("Waiting for Layer2Meta from graph_sync bridge; skip policy inference to avoid wrong world goals")
                self._last_no_data_log_ts = now
            self._set_profile(
                total_ms=0.0,
                map_ms=0.0,
                rebuild_ms=0.0,
                feature_ms=0.0,
                infer_ms=0.0,
                goal_ms=0.0,
                exit_reason="no_meta",
            )
            return
        if (
            self.map_input_mode == "graph_sync"
            and getattr(self, "startup_require_graph_ready", False)
            and not getattr(self, "_startup_graph_ready", False)
        ):
            gate = self._graph_startup_gate_info()
            if not gate["ready"]:
                now = time.monotonic()
                if (now - self._last_no_data_log_ts) > 2.0:
                    self.get_logger().warn(
                        "Waiting for graph readiness before first action: "
                        f"topo_updates={gate['topo_updates']}/{gate['need_topo_updates']}, "
                        f"frontiers={gate['frontiers']}/{gate['need_frontiers']}, "
                        f"viewpoints={gate['viewpoints']}/{gate['need_viewpoints']}, "
                        f"free_cells={gate['free_cells']}/{gate['need_free_cells']}, "
                        f"min_local_free={(gate['min_local_free'] if gate['min_local_free'] is not None else -1)}/{gate['need_local_free']}"
                    )
                    self._last_no_data_log_ts = now
                self._append_event_log(
                    "startup_wait_graph",
                    topo_updates=int(gate["topo_updates"]),
                    need_topo_updates=int(gate["need_topo_updates"]),
                    frontiers=int(gate["frontiers"]),
                    need_frontiers=int(gate["need_frontiers"]),
                    viewpoints=int(gate["viewpoints"]),
                    need_viewpoints=int(gate["need_viewpoints"]),
                    free_cells=int(gate["free_cells"]),
                    need_free_cells=int(gate["need_free_cells"]),
                    configured_need_free_cells=int(gate["configured_need_free_cells"]),
                    min_local_free=int(gate["min_local_free"]) if gate["min_local_free"] is not None else -1,
                    need_local_free=int(gate["need_local_free"]),
                )
                self._set_profile(
                    total_ms=0.0,
                    map_ms=0.0,
                    rebuild_ms=0.0,
                    feature_ms=0.0,
                    infer_ms=0.0,
                    goal_ms=0.0,
                    exit_reason="startup_wait_graph",
                )
                return
            self._startup_graph_ready = True
            self._append_event_log(
                "startup_graph_ready",
                topo_updates=int(gate["topo_updates"]),
                frontiers=int(gate["frontiers"]),
                viewpoints=int(gate["viewpoints"]),
                free_cells=int(gate["free_cells"]),
                need_free_cells=int(gate["need_free_cells"]),
                configured_need_free_cells=int(gate["configured_need_free_cells"]),
                min_local_free=int(gate["min_local_free"]) if gate["min_local_free"] is not None else -1,
            )
            self.get_logger().info(
                "graph startup gate passed: "
                f"topo_updates={gate['topo_updates']}, "
                f"frontiers={gate['frontiers']}, viewpoints={gate['viewpoints']}, "
                f"free_cells={gate['free_cells']}, need_free_cells={gate['need_free_cells']}, "
                f"min_local_free={(gate['min_local_free'] if gate['min_local_free'] is not None else -1)}"
            )
        total_t0 = time.perf_counter()
        self.map_update_flag = False

        if self.map_input_mode == "graph_sync" and not getattr(self, "_graph_snapshot_committed_once", False):
            self._commit_live_graph_snapshot("bootstrap_replan")

        # Seed committed/predictive discrete poses from real odom before any goal exists.
        for _i, _uav_key in enumerate(self.uav_odoms.keys()):
            if _i < len(self.graph_pose_initialized) and not self.graph_pose_initialized[_i]:
                seed_rc = [
                    int(self.uav_odoms[_uav_key][0]),
                    int(self.uav_odoms[_uav_key][1]),
                ]
                self.graph_pose_committed_ds[_i] = list(seed_rc)
                self.graph_pose_ds[_i] = list(seed_rc)
                self.active_goal_ds[_i] = list(seed_rc)
                self.graph_pose_initialized[_i] = True

        # 计算方框的边界
        # 当需要遍历所有 UAV 时
        self.frontier_remove_list = []
        before_map_processing = time.perf_counter()
        if self.map_input_mode == "graph_sync":
            self._rebuild_frontiers_from_graph()
        else:
            for uav_id in self.uav_odoms:
                self.get_logger().debug(f'uav_id: {uav_id}, uav_odom: {self.uav_odoms[uav_id]}')
                odom_x, odom_y = self.uav_odoms[uav_id]
                radius = self.viewrange 
                left = int(max(0, odom_y - radius))
                right = int(min(self.width, odom_y + radius + 1))
                bottom = int(max(0, odom_x - radius))
                top = int(min(self.height, odom_x + radius + 1)) 

                # 创建局部坐标系
                y, x = np.ogrid[left:right, bottom:top,]
                # 计算到中心的距离
                dist_from_center = np.sqrt((x - odom_x)**2 + (y - odom_y)**2)
     
                mask = dist_from_center < self.viewrange + 1 
                y_indices, x_indices = np.nonzero(mask)
                # 在有效的点上进行循环
                for i in range(len(y_indices)):
                    y_local = y_indices[i]
                    x_local = x_indices[i]
                    # 转换回全局坐标
                    y_global = y_local + left 
                    x_global = x_local + bottom
                    # 这里似乎有问题
                    if x_global >= self.map_size_height or y_global >= self.map_size_width:
                        # self.get_logger().info(f'{Fore.RED}Out of map{Style.RESET_ALL}')
                        continue
                    # 在这里执行你想要的操作
                    self.update_frontier(self.downsampled_map_data, x_global, y_global,odom_x, odom_y) 
            self.get_logger().debug(f'frontier_dict size: {len(self.frontier_dict)}')
            delete_frontier_set = set(self.frontier_remove_list)
            for nx, ny in delete_frontier_set:
                if (nx, ny) in self.frontier_dict:
                    del self.frontier_dict[(nx, ny)]
            self.frontier_node_feature = [
                [nx, ny, int(val[-1]), 0, 0, self.node_type[0]]
                for (nx, ny), val in self.frontier_dict.items()
            ]
        
        # for         
        
        if self.map_input_mode == "graph_sync":
            for uav_id in self.uav_odoms:
                self.last_uav_odoms[uav_id] = self.uav_odoms[uav_id]
        else:
            for uav_id in self.uav_odoms:
                self.update_viewpoints(uav_id, self.uav_odoms[uav_id][0], self.uav_odoms[uav_id][1])   
                self.last_uav_odoms[uav_id] = self.uav_odoms[uav_id] 
        self.delete_view_point() # 只要一次就够
        after_map_processing = time.perf_counter()
        rebuild_ms = 1000.0 * (after_map_processing - before_map_processing)


        # robot node feature
        # 根据环境制作当前的动态图, 核心代码
        before_feature_processing = time.perf_counter()
        self.other_robot_node_feature = [[] for i in range(self.drone_num)]
        for i, ego_uav_id in enumerate(self.uav_odoms.keys()):
            ego_r, ego_c = self._effective_pos_for_graph(i, ego_uav_id)
            self.robot_node_feature[i] = [ego_r, ego_c, 0, 0, 0, self.node_type[2]]
            for j, each_uav_id in enumerate(self.uav_odoms):
                if each_uav_id != ego_uav_id:
                    other_r, other_c = self._effective_pos_for_graph(j, each_uav_id)
                    to_distance_x = other_r - ego_r
                    to_distance_y = other_c - ego_c
                    to_distance_x = math.copysign(min(abs(to_distance_x), self.classification_range), to_distance_x)
                    to_distance_y = math.copysign(min(abs(to_distance_y), self.classification_range), to_distance_y)
                    self.other_robot_node_feature[i].append([other_r, other_c, 0, to_distance_x, to_distance_y, self.node_type[3]])
            self.joint_robot_node_feature[i] = [ego_r, ego_c, 0, 0, 0, self.node_type[2]]
        # 安全检测，防止没有处理好 frontier 导致下面代码出错
        if not self.frontier_dict:
            now = time.monotonic()
            if (now - self._last_no_frontier_log_ts) > 2.0:
                self.get_logger().warn(
                    f"No frontier available (mode={self.map_input_mode}, viewrange={self.viewrange} cells), skip policy inference"
                )
                self._last_no_frontier_log_ts = now
            self._append_event_log(
                "no_frontier",
                mode=self.map_input_mode,
                viewrange_cells=int(self.viewrange),
                frontier_count=int(len(self.frontier_dict)),
                viewpoint_count=int(len(self.viewpoint_dict)),
            )
            self._freeze_all_goals(reason="no_frontier")
            self.map_update_flag = True
            total_t1 = time.perf_counter()
            self._set_profile(
                total_ms=1000.0 * (total_t1 - total_t0),
                map_ms=1000.0 * (after_map_processing - before_map_processing),
                rebuild_ms=rebuild_ms,
                feature_ms=1000.0 * (time.perf_counter() - before_feature_processing),
                infer_ms=0.0,
                goal_ms=0.0,
                exit_reason="no_frontier",
            )
            return 
        # self.graph_lisst, joint_graph_data, indices_viewpoint, self.robot_positions[:, :2], no_viewpoint = self.node_feature_process(self.frontier_node_feature, self.viewpoint_node_feature, self.robot_node_feature, self.other_robot_node_feature, self.joint_robot_node_feature)
        if not any(self.viewpoint_node_feature):
            done = [True for i in range(self.drone_num)]    
            self.map_update_flag = True
            self.get_logger().warn("No viewpoint available, skip policy inference for this cycle")
            self._append_event_log(
                "no_viewpoint",
                frontier_count=int(len(self.frontier_dict)),
                viewpoint_count=int(len(self.viewpoint_dict)),
                drone_num=int(self.drone_num),
            )
            self._freeze_all_goals(reason="no_viewpoint")
            total_t1 = time.perf_counter()
            self._set_profile(
                total_ms=1000.0 * (total_t1 - total_t0),
                map_ms=1000.0 * (after_map_processing - before_map_processing),
                rebuild_ms=rebuild_ms,
                feature_ms=1000.0 * (time.perf_counter() - before_feature_processing),
                infer_ms=0.0,
                goal_ms=0.0,
                exit_reason="no_viewpoint",
            )
            return                                 
        else: 
            self.graph_list, joint_graph_data, indices_viewpoint, self.robot_positions, no_viewpoint = self.node_feature_process(self.frontier_node_feature, self.viewpoint_node_feature, self.robot_node_feature, self.other_robot_node_feature, self.joint_robot_node_feature)
        after_feature_processing = time.perf_counter()
        feature_ms = 1000.0 * (after_feature_processing - before_feature_processing)
        if no_viewpoint:
            done = [True for i in range(self.drone_num)]
        # self.get_logger().info(f'{Fore.GREEN}indices_viewpoint: {indices_viewpoint}{Style.RESET_ALL}')
        before_time = time.perf_counter()
        
        with torch.no_grad():
            if self.inference_backend == "pt_shared":
                if self.shared_actor is None:
                    raise RuntimeError("shared_actor is not initialized")

                indices_viewpoint_batch, all_indices_viewpoint_batch = self._build_index_batches()

                actions, action_log_probs, ratio = self.shared_actor(
                    self.graph_list,
                    indices_viewpoint_batch,
                    all_indices_viewpoint_batch,
                    deterministic=self.pt_deterministic,
                )

                for i in range(self.drone_num):
                    action_idx = int(actions[i].item()) if i < actions.shape[0] else 0
                    self.action_list[i] = np.array([[action_idx]], dtype=np.int64)

                self.get_logger().debug(
                    f'{Fore.GREEN}pt_shared actions={actions.squeeze(-1).tolist()} logp_shape={list(action_log_probs.shape)}{Style.RESET_ALL}'
                )
            elif self.inference_backend == "onnx_shared_dynamic":
                if self.shared_actor is None or self.shared_dynamic_onnx_policy is None:
                    raise RuntimeError("onnx_shared_dynamic backend is not initialized")

                indices_viewpoint_batch, all_indices_viewpoint_batch = self._build_index_batches()
                actions, mixed_probs, mix_ratio = self.shared_dynamic_onnx_policy.infer_from_graphs(
                    shared_actor=self.shared_actor,
                    graph_list=self.graph_list,
                    indices_viewpoint_batch=indices_viewpoint_batch,
                    all_indices_viewpoint_batch=all_indices_viewpoint_batch,
                    deterministic=self.pt_deterministic,
                )
                actions = self._coordinate_joint_actions(
                    mixed_probs=mixed_probs,
                    indices_viewpoint_batch=indices_viewpoint_batch,
                    all_indices_viewpoint_batch=all_indices_viewpoint_batch,
                )
                self._append_event_log(
                    "joint_coordination",
                    backend="onnx_shared_dynamic",
                    actions=[int(actions[i][0]) for i in range(min(self.drone_num, actions.shape[0]))],
                    neighbor_counts=[len(v) for v in indices_viewpoint_batch],
                    full_counts=[len(v) for v in all_indices_viewpoint_batch],
                    neighbor_indices=[[int(x) for x in v[:16]] for v in indices_viewpoint_batch],
                    full_indices=[[int(x) for x in v[:16]] for v in all_indices_viewpoint_batch],
                    neighbor_coords=self._indices_batch_to_coords(indices_viewpoint_batch),
                    full_coords=self._indices_batch_to_coords(all_indices_viewpoint_batch),
                )

                for i in range(self.drone_num):
                    action_idx = int(actions[i][0]) if i < actions.shape[0] else 0
                    self.action_list[i] = np.array([[action_idx]], dtype=np.int64)

                self.get_logger().debug(
                    f'{Fore.GREEN}onnx_shared_dynamic actions={actions.reshape(-1).tolist()} probs_shape={list(mixed_probs.shape)} mix_shape={list(mix_ratio.shape)}{Style.RESET_ALL}'
                )
            elif self.inference_backend == "onnx_full_dynamic":
                if self.full_dynamic_onnx_policy is None:
                    raise RuntimeError("onnx_full_dynamic backend is not initialized")

                indices_viewpoint_batch, all_indices_viewpoint_batch = self._build_index_batches()
                actions, mixed_probs, mix_ratio = self.full_dynamic_onnx_policy.infer_from_graphs(
                    graph_list=self.graph_list,
                    indices_viewpoint_batch=indices_viewpoint_batch,
                    all_indices_viewpoint_batch=all_indices_viewpoint_batch,
                    deterministic=self.pt_deterministic,
                )
                actions = self._coordinate_joint_actions(
                    mixed_probs=mixed_probs,
                    indices_viewpoint_batch=indices_viewpoint_batch,
                    all_indices_viewpoint_batch=all_indices_viewpoint_batch,
                )
                self._append_event_log(
                    "joint_coordination",
                    backend="onnx_full_dynamic",
                    actions=[int(actions[i][0]) for i in range(min(self.drone_num, actions.shape[0]))],
                    neighbor_counts=[len(v) for v in indices_viewpoint_batch],
                    full_counts=[len(v) for v in all_indices_viewpoint_batch],
                    neighbor_indices=[[int(x) for x in v[:16]] for v in indices_viewpoint_batch],
                    full_indices=[[int(x) for x in v[:16]] for v in all_indices_viewpoint_batch],
                    neighbor_coords=self._indices_batch_to_coords(indices_viewpoint_batch),
                    full_coords=self._indices_batch_to_coords(all_indices_viewpoint_batch),
                )

                for i in range(self.drone_num):
                    action_idx = int(actions[i][0]) if i < actions.shape[0] else 0
                    self.action_list[i] = np.array([[action_idx]], dtype=np.int64)

                self.get_logger().debug(
                    f'{Fore.GREEN}onnx_full_dynamic actions={actions.reshape(-1).tolist()} probs_shape={list(mixed_probs.shape)} mix_shape={list(mix_ratio.shape)}{Style.RESET_ALL}'
                )
            else:
                raise RuntimeError(
                    f"Unsupported inference_backend={self.inference_backend}. "
                    "Only pt_shared, onnx_shared_dynamic and onnx_full_dynamic are available."
                )

        after_time = time.perf_counter()
        goal_t0 = time.perf_counter()
        self.last_goal = next_uav_odom_list = self.get_goal() # 获取uav们在下一个时间步的中在downsampled map中要去的位置
        goal_t1 = time.perf_counter()
        self.map_update_flag = True
        total_t1 = time.perf_counter()
        self._set_profile(
            total_ms=1000.0 * (total_t1 - total_t0),
            map_ms=1000.0 * (after_map_processing - before_map_processing),
            rebuild_ms=rebuild_ms,
            feature_ms=feature_ms,
            infer_ms=1000.0 * (after_time - before_time),
            goal_ms=1000.0 * (goal_t1 - goal_t0),
            exit_reason="ok",
        )
        self.get_logger().debug(f'goal_of_drone: {self.goal_of_drone}')
        return next_uav_odom_list
