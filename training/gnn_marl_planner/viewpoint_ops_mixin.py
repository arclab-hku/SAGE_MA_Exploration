from __future__ import annotations

import copy
import math
import numpy as np
import torch
from collections import defaultdict
from colorama import Fore, Style
from .uav_index import uav_key_from_index

class ViewpointOpsMixin:
    def _graph_sync_anchor_rc(self, drone_id: int):
        uav_key = uav_key_from_index(self.uav_odoms, drone_id)
        if uav_key is None:
            return None
        if hasattr(self, "_effective_pos_for_graph"):
            try:
                anchor_r, anchor_c = self._effective_pos_for_graph(drone_id, uav_key)
                return int(anchor_r), int(anchor_c)
            except Exception:
                pass
        return int(self.uav_odoms[uav_key][0]), int(self.uav_odoms[uav_key][1])

    def _local_neighbor_candidates(self, center_r: int, center_c: int):
        min_x = max(0, center_r - 1)
        min_y = max(0, center_c - 1)
        max_x = min(self.map_size_height - 1, center_r + 1)
        max_y = min(self.map_size_width - 1, center_c + 1)
        return [
            (min_x, center_c), (max_x, center_c),
            (center_r, min_y), (center_r, max_y),
        ]

    def update_frontier(self, data, x, y, odom_x, odom_y):
        # odom_x, odom_y是无人机的位置, 用于辅助判断哪些点是刚好在边缘的
        # 防止frontier的检测超出地图的边界
        min_x = max(0, x-1)
        min_y = max(0, y-1)
        max_x = min(self.map_size_height-1, x+1)
        max_y = min(self.map_size_width-1, y+1)
        # self.get_logger().info(f'max_x: {max_x}, min_x: {min_x}, max_y: {max_y}, min_y: {min_y}')
        # self.get_logger().info(f'x: {x}, y: {y}')
        neighbors = [(min_x, y), (max_x, y), (x, min_y), (x, max_y)]
        utility = 0
        free_area = False
        unknown_area = False

        for nx, ny in neighbors:
            # 查看是否符合条件1： neighbor中的点有一个是已知区域
            if data[nx, ny] == 0:
                free_area = True
            # 查看是否符合条件2： neighbor中的点有一个是未知区域
            if data[nx, ny] == -1:
                unknown_area = True
            if free_area and unknown_area:
                break

        utility = int(free_area and unknown_area)

        if utility > 0:
            # check_points = self.ray_point(nx, ny, odom_x, odom_y)
            # for point in check_points:
            #     # self.get_logger().info(f'{Fore.GREEN}point: {point}{Style.RESET_ALL}')
            #     if self.downsampled_map_data[point[0], point[1]] == 100:
            #         return
            
            # self.get_logger().info(f'{Fore.GREEN}Have frontier!{Style.RESET_ALL}')
            # for nx, ny in neighbors:
            # 如果neighbor中的点有一个是未知区域，那么在frontier这个字典中添加这个neighbor点
            if data[x, y] == -1:  
                # 边界点的特征[点的种类编号, 周围有多少个未知区域]
                self.frontier_dict[(x, y)] = [self.node_type[0], utility]
                self.max_frontier_x = max(x, self.max_frontier_x)
                self.min_frontier_x = min(x, self.min_frontier_x)
                self.max_frontier_y = max(y, self.max_frontier_y)
                self.min_frontier_y = min(y, self.min_frontier_y)
                self.frontier_node_feature.append([x, y, utility, 0, 0, self.node_type[0]])
                
            # 排除不属于边点
            elif (x, y) in self.frontier_dict:
                self.frontier_remove_list.append((x, y))
                # self.frontier_node_feature.remove([nx, ny, self.frontier_dict[(nx, ny)][-1], self.node_type[0]])
                # del self.frontier_dict[(nx, ny)]
                if self.frontier_dict:
                    if self.max_frontier_x == x:
                        self.max_frontier_x = max([key[0] for key in self.frontier_dict.keys()])                            
                    if self.min_frontier_x == x:
                        self.min_frontier_x = min([key[0] for key in self.frontier_dict.keys()])
                    if self.max_frontier_y == y:
                        self.max_frontier_y = max([key[1] for key in self.frontier_dict.keys()])
                    if self.min_frontier_y == y:
                        self.min_frontier_y = min([key[1] for key in self.frontier_dict.keys()])
                else:
                    self.max_frontier_x = 0
                    self.min_frontier_x = 0
                    self.max_frontier_y = 0
                    self.min_frontier_y = 0
        else:
            # 如果neighbor中的点都是已经探索过了，那么在frontier这个字典中检查是否有这个neighbor点，有的话就删除
            if (x, y) in self.frontier_dict:
                self.frontier_remove_list.append((x, y))
                # self.frontier_node_feature.remove([nx, ny, self.frontier_dict[(nx, ny)][-1], 0, 0, self.node_type[0]])
                # del self.frontier_dict[(nx, ny)]
                if self.frontier_dict:
                    if self.max_frontier_x == x:
                        self.max_frontier_x = max([key[0] for key in self.frontier_dict.keys()])                            
                    if self.min_frontier_x == x:
                        self.min_frontier_x = min([key[0] for key in self.frontier_dict.keys()])
                    if self.max_frontier_y == y:
                        self.max_frontier_y = max([key[1] for key in self.frontier_dict.keys()])
                    if self.min_frontier_y == y:
                        self.min_frontier_y = min([key[1] for key in self.frontier_dict.keys()])
                else:
                    self.max_frontier_x = 0
                    self.min_frontier_x = 0
                    self.max_frontier_y = 0
                    self.min_frontier_y = 0

    def ray_point(self,x0, y0, x1, y1):
        """Vectorized ray tracing from (x0,y0) to (x1,y1)"""
        num_steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
        t = np.linspace(0, 1, num_steps)
        x = np.round(x0 * (1 - t) + x1 * t).astype(int)
        y = np.round(y0 * (1 - t) + y1 * t).astype(int)
        return np.column_stack((x, y))

    def update_viewpoints(self, uav_id, x, y):
        # update view_points
        min_x = max(0, x-1)
        min_y = max(0, y-1)
        max_x = min(self.map_size_height-1, x+1)
        max_y = min(self.map_size_width-1, y+1)
        # ii, jj = np.mgrid[min_x:max_x+1, min_y:max_y+1]
        # candidate_view_points = np.stack([ii.ravel(), jj.ravel()], axis=-1)
        candidate_view_points = [
            (min_x, y), (max_x, y), (x, min_y), (x, max_y),
            (min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y),
        ]
        # self.get_logger().info(f'candidate_view_points: {candidate_view_points}')
        # Find unknown grids
        have_viewpoint = False
        for nx, ny in candidate_view_points:
            view_min_x = max(0, nx-self.viewrange)
            view_min_y = max(0, ny-self.viewrange)
            view_max_x = min(self.map_size_height, nx+self.viewrange + 1)
            view_max_y = min(self.map_size_width, ny+self.viewrange + 1)
            # self.get_logger().info(f'{Fore.GREEN}view_min_x: {view_min_x}, view_min_y: {view_min_y}, view_max_x: {view_max_x}, view_max_y: {view_max_y}{Style.RESET_ALL}')
            ii, jj = np.mgrid[view_min_x:view_max_x, view_min_y:view_max_y]
            view_points_view_range = np.stack([ii.ravel(), jj.ravel()], axis=-1)
            # how many frontiers can be observed by this view point
            utility = 0
            frontier_coordinate_list = []
            # 先提取视野信息
            for vx, vy in view_points_view_range:   
                # if (vx-nx)**2 + (vy-ny)**2 > (self.viewrange+2)**2:
                #     continue   
                if (vx, vy) in self.frontier_dict and self.downsampled_map_data[vx, vy] == -1:
                    frontier_coordinate_list.append((int(vx), int(vy)))
                    utility += 1

            # if frontier_coordinate_list:
                # for frontier_x, frontier_y in frontier_coordinate_list:
                #     check_points = self.ray_point(frontier_x, frontier_y, nx, ny)
                #     for point in check_points:
                #         # self.get_logger().info(f'{Fore.GREEN}point: {point}{Style.RESET_ALL}')
                #         if self.downsampled_map_data[point[0], point[1]] == 100:
                #             while (frontier_x, frontier_y) in frontier_coordinate_list:
                #                 frontier_coordinate_list.remove((frontier_x, frontier_y))
                #             utility -= 1
                #             break
            if frontier_coordinate_list:
                have_viewpoint = True
                self.max_viewpoint_x = max(nx, self.max_viewpoint_x)
                self.min_viewpoint_x = min(nx, self.min_viewpoint_x)
                self.max_viewpoint_y = max(ny, self.max_viewpoint_y)
                self.min_viewpoint_y = min(ny, self.min_viewpoint_y)
                self.viewpoint_dict[(nx, ny)] = [self.node_type[1], utility, frontier_coordinate_list]
                # 判断边界点是否在以 view_range为半径的
                # 无正负关系的
                # to_drone_distance_x = min(abs(nx - drone.pos[0]), self.classification_range)
                # to_drone_distance_y = min(abs(ny - drone.pos[1]), self.classification_range)
                # 有正负关系的
                for drone_id in range(self.drone_num):
                    drone_key = uav_key_from_index(self.uav_odoms, drone_id)
                    if drone_key is None:
                        continue
                    # 重要修改
                    to_drone_distance_x = nx - self.uav_odoms[drone_key][0]
                    to_drone_distance_y = ny - self.uav_odoms[drone_key][1]
                    to_drone_distance_x = math.copysign(min(abs(to_drone_distance_x), self.classification_range), to_drone_distance_x)
                    to_drone_distance_y = math.copysign(min(abs(to_drone_distance_y), self.classification_range), to_drone_distance_y)
                    self.viewpoint_node_feature[drone_id][(nx, ny)]=[nx, ny, utility, to_drone_distance_x, to_drone_distance_y, self.node_type[1]]
        if not have_viewpoint:
            if self.map_input_mode == "graph_sync":
                # In graph-sync mode we avoid hidden fallback behavior.
                self.astar_target[uav_id] = None
                return
            self.get_logger().debug(f'{Fore.RED}No 1-hop viewpoint found, try A* fallback{Style.RESET_ALL}')
            from .astar import astar
            path = astar(self.downsampled_map_data, (self.uav_odoms[uav_id][0], self.uav_odoms[uav_id][1]))
            # 依次判断
            self.get_logger().info(f'{Fore.GREEN}path: {path}{Style.RESET_ALL}')
            if path:
                self.get_logger().info(f'{Fore.GREEN}path: {path}{Style.RESET_ALL}')
                    # time.sleep(1000000)
                # 更新视野
                for (nx, ny) in path:
                    view_min_x = max(0, nx-self.viewrange + 1)
                    view_min_y = max(0, ny-self.viewrange + 1)
                    view_max_x = min(self.map_size_height-1, nx+self.viewrange)
                    view_max_y = min(self.map_size_width-1, ny+self.viewrange)
                    # self.get_logger().info(f'{Fore.GREEN}view_min_x: {view_min_x}, view_min_y: {view_min_y}, view_max_x: {view_max_x}, view_max_y: {view_max_y}{Style.RESET_ALL}')
                    ii, jj = np.mgrid[view_min_x:view_max_x, view_min_y:view_max_y]
                    view_points_view_range = np.stack([ii.ravel(), jj.ravel()], axis=-1)
                    # how many frontiers can be observed by this view point
                    utility = 0
                    frontier_coordinate_list = []
                    # 先提取视野信息
                    for vx, vy in view_points_view_range:   
                        if (vx-nx)**2 + (vy-ny)**2 > self.viewrange**2:
                            continue   
                        if (vx, vy) in self.frontier_dict:
                            # self.get_logger().info(f'{Fore.GREEN}view point: {vx, vy}{Style.RESET_ALL}')
                            frontier_coordinate_list.append((int(vx), int(vy)))
                            utility += 1                    
                    if frontier_coordinate_list:
                        for frontier_x, frontier_y in frontier_coordinate_list:
                            check_points = self.ray_point(frontier_x, frontier_y, nx, ny)
                            for point in check_points:
                                # self.get_logger().info(f'{Fore.GREEN}point: {point}{Style.RESET_ALL}')
                                if self.downsampled_map_data[point[0], point[1]] == 100:
                                    while (frontier_x, frontier_y) in frontier_coordinate_list:
                                        frontier_coordinate_list.remove((frontier_x, frontier_y))
                                    utility -= 1
                                    break

                    if frontier_coordinate_list:
                        have_viewpoint = True
                        self.max_viewpoint_x = max(nx, self.max_viewpoint_x)
                        self.min_viewpoint_x = min(nx, self.min_viewpoint_x)
                        self.max_viewpoint_y = max(ny, self.max_viewpoint_y)
                        self.min_viewpoint_y = min(ny, self.min_viewpoint_y)
                        self.viewpoint_dict[(nx, ny)] = [self.node_type[1], utility, frontier_coordinate_list]
                        # 判断边界点是否在以 view_range为半径的
                        # 无正负关系的
                        # to_drone_distance_x = min(abs(nx - drone.pos[0]), self.classification_range)
                        # to_drone_distance_y = min(abs(ny - drone.pos[1]), self.classification_range)
                        # 有正负关系的
                        for drone_id in range(self.drone_num):
                            drone_key = uav_key_from_index(self.uav_odoms, drone_id)
                            if drone_key is None:
                                continue
                            to_drone_distance_x = nx - self.uav_odoms[drone_key][0]
                            to_drone_distance_y = ny - self.uav_odoms[drone_key][1]
                            to_drone_distance_x = math.copysign(min(abs(to_drone_distance_x), self.classification_range), to_drone_distance_x)
                            to_drone_distance_y = math.copysign(min(abs(to_drone_distance_y), self.classification_range), to_drone_distance_y)
                            self.viewpoint_node_feature[drone_id][(nx, ny)]=[nx, ny, utility, to_drone_distance_x, to_drone_distance_y, self.node_type[1]]
                        self.astar_target[uav_id] = (nx, ny)
                        break
                else:
                    self.astar_target[uav_id] = None

    def delete_view_point(self):
        # In graph_sync mode, viewpoints are authoritative from ROS1 graph.
        # Do not run local hard-pruning here; only compute neighbor matching.
        if self.map_input_mode == "graph_sync":
            viewpoint_keys = sorted(self.viewpoint_dict.keys())
            viewpoint_index_map = {coord: idx for idx, coord in enumerate(viewpoint_keys)}
            # Align with training semantics:
            # - all_indices_viewpoint: full viewpoint pool for each UAV
            # - matching_indices: ego-local neighbor subset (used as neighbor mask / type-4 mark)
            self.all_indices_viewpoint = [
                list(range(len(viewpoint_keys))) for _ in range(self.drone_num)
            ]
            self.matching_indices = [[] for _ in range(self.drone_num)]

            for drone_id in range(self.drone_num):
                anchor_rc = self._graph_sync_anchor_rc(drone_id)
                if anchor_rc is None:
                    continue
                anchor_r, anchor_c = anchor_rc

                # Training/runtime should share the same local-action meaning:
                # first try the exact 1-hop neighborhood around the discrete graph pose.
                for candidate in self._local_neighbor_candidates(anchor_r, anchor_c):
                    idx = viewpoint_index_map.get(candidate)
                    if idx is None:
                        continue
                    self.matching_indices[drone_id].append(idx)

                # Preserve full-pool ordering while exposing a local subset mask.
                seen = set()
                ordered = []
                for idx in self.matching_indices[drone_id]:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    ordered.append(idx)
                self.matching_indices[drone_id] = ordered

                for idx in self.matching_indices[drone_id]:
                    vx, vy = viewpoint_keys[idx]
                    if (vx, vy) in self.viewpoint_node_feature[drone_id]:
                        self.viewpoint_node_feature[drone_id][(vx, vy)][-1] = self.node_type[4]
                    if (vx, vy) in self.viewpoint_dict:
                        self.viewpoint_dict[(vx, vy)][0] = self.node_type[4]
            return

        self.delete_num = 0
        self.delete_item = []
        for drone_id in range(self.drone_num):
            # 首先把无人机上一次的位置放进去
            drone_key = uav_key_from_index(self.last_uav_odoms, drone_id)
            if drone_key is not None:
                self.delete_item.append(self.last_uav_odoms[drone_key])
                self.get_logger().debug(f'delete last uav odometry: {self.last_uav_odoms[drone_key]}')
        for (nx, ny), (node_type, utility, frontier_coordinate_list) in self.viewpoint_dict.items():
            # how many frontiers can be observed by this view point
            utility = 0
            exist_frontier = False
            for vx, vy in frontier_coordinate_list:
                unknown = False
                block = False
                # 如果是未知点
                if self.downsampled_map_data[vx, vy] == -1:
                    unknown = True
                    # 如果这个未知点到viewpoint的连线上没有障碍物                    
                    check_points = self.ray_point(vx, vy, nx, ny)
                    for point in check_points:
                        if self.downsampled_map_data[point[0], point[1]] == 100:
                            block = True
                            break
                if unknown and not block:
                    utility += 1
                    exist_frontier = True
            # 这里应该更新每个viewpoint的utility
            self.viewpoint_dict[(nx, ny)][1] = utility
            for drone_id in range(self.drone_num):
                drone_key = uav_key_from_index(self.uav_odoms, drone_id)
                if drone_key is None:
                    continue
                # 然后检测其他的viewpoint是否已经失效了
                self.viewpoint_node_feature[drone_id][(nx, ny)][2] = utility
                # 判断边界点是否在以 view_range为半径的
                # to_drone_distance_x = min(abs(nx - self.uav_odoms[temp_uav_id][0]), self.classification_range)
                # to_drone_distance_y = min(abs(ny - self.uav_odoms[temp_uav_id][1]), self.classification_range)
                # 考虑到有正负关系
                to_drone_distance_x = nx - self.uav_odoms[drone_key][0]
                to_drone_distance_y = ny - self.uav_odoms[drone_key][1]
                to_drone_distance_x = math.copysign(min(abs(to_drone_distance_x), self.classification_range), to_drone_distance_x)
                to_drone_distance_y = math.copysign(min(abs(to_drone_distance_y), self.classification_range), to_drone_distance_y)
                # self.frontier_node_feature.append([nx, ny, utility, to_drone_distance_x, to_drone_distance_y, self.node_type[0]])
                self.viewpoint_node_feature[drone_id][(nx, ny)]=[nx, ny, utility, to_drone_distance_x, to_drone_distance_y, self.node_type[1]]

            if not exist_frontier:
                self.delete_num += 1
                self.delete_item.append((nx, ny))
            else:  
                neighbor_list = [(nx-1, ny), (nx+1, ny), (nx, ny-1), (nx, ny+1)]
                for neighbor in neighbor_list:
                    if self.downsampled_map_data[neighbor[0], neighbor[1]] == 100:
                        self.delete_num += 1
                        self.delete_item.append((nx, ny))
                        break
        self.delete_item = list(set(self.delete_item))   
        self.get_logger().debug(f'len(self.delete_item): {len(self.delete_item)}')
        for delete_coordinate in self.delete_item: 
            if (delete_coordinate[0], delete_coordinate[1]) not in self.viewpoint_node_feature[0]:
                # self.delete_item.remove(delete_coordinate)
                continue
            del self.viewpoint_dict[(delete_coordinate[0], delete_coordinate[1])]
            for drone_id in range(self.drone_num):
                del self.viewpoint_node_feature[drone_id][(delete_coordinate[0], delete_coordinate[1])]    
            if self.viewpoint_dict:
                if self.max_viewpoint_x == nx:
                    self.max_viewpoint_x = max([key[0] for key in self.viewpoint_dict.keys()])
                if self.min_viewpoint_x == nx:
                    self.min_viewpoint_x = min([key[0] for key in self.viewpoint_dict.keys()])
                if self.max_viewpoint_y == ny:
                    self.max_viewpoint_y = max([key[1] for key in self.viewpoint_dict.keys()])
                if self.min_viewpoint_y == ny:
                    self.min_viewpoint_y = min([key[1] for key in self.viewpoint_dict.keys()])
            else:
                self.max_viewpoint_x = 0
                self.min_viewpoint_x = 0
                self.max_viewpoint_y = 0
                self.min_viewpoint_y = 0
        viewpoint_keys = sorted(self.viewpoint_dict.keys())
        viewpoint_index_map = {coord: idx for idx, coord in enumerate(viewpoint_keys)}
        self.matching_indices = [[] for i in range(self.drone_num)]
        for i, uav_id in enumerate(self.uav_odoms):
            min_x = max(0, self.uav_odoms[uav_id][0] -1)
            min_y = max(0,  self.uav_odoms[uav_id][1]-1)
            max_x = min(self.map_size_height,  self.uav_odoms[uav_id][0]+1)
            max_y = min(self.map_size_width,  self.uav_odoms[uav_id][1]+1)
            # ii, jj = np.mgrid[min_x:max_x+1, min_y:max_y+1]
            # candidate_view_points = np.stack([ii.ravel(), jj.ravel()], axis=-1)
            candidate_view_points = [
                (min_x, self.uav_odoms[uav_id][1]), (max_x, self.uav_odoms[uav_id][1]),
                (self.uav_odoms[uav_id][0], min_y), (self.uav_odoms[uav_id][0], max_y),
                (min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y),
            ]
            for (nx, ny) in candidate_view_points:
                if (nx, ny) in self.viewpoint_dict:
                    index = viewpoint_index_map[(nx, ny)]
                    self.matching_indices[i].append(index)
                    self.viewpoint_node_feature[i][(nx, ny)][-1] = self.node_type[4]
                    self.viewpoint_dict[(nx, ny)][0] = self.node_type[4]
