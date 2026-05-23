import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
import numpy as np
from matplotlib import pyplot as plt
from colorama import Fore, Style
import torch
from collections import defaultdict
import copy
from typing import List, Tuple
from itertools import product
from collections import deque
from torch_geometric.data import Data
from itertools import islice
import re

class occupancy_to_graph(Node):
    def __init__(self):
        super().__init__('occupancy_to_graph')
        # 订阅odom话题
        self.discover_and_subscribe()
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/sdf_map/occupancy_grid_2d_1',
            self.listener_callback,
            10)
        self.uav_id = '1'
        self.subscription  # prevent unused variable warning
        
        self.map_origin = [7, 15]   # 单位为m，sdf地图的中心点
        self.width = None
        self.height = None
        self.resolution = 0.1
        self.map_size_height = int(2 * self.map_origin[1] / self.resolution)
        self.map_size_width = int(2 * self.map_origin[0] / self.resolution)

        self.downsample_factor = 1  # 降采样因子
        self.map_data = None # occupied: 100, free: 0, unknown: -1

        # 无人机参数
        self.viewrange = 5 / self.resolution  # 视野半径，单位为m
        self.uav_odoms = {}
        # 拓扑图相关的参数

        self.min_frontier_x = float('inf')
        self.max_frontier_x = 0
        self.min_frontier_y = float('inf')
        self.max_frontier_y = 0
        self.last_min_frontier_x = float('inf')
        self.last_max_frontier_x = 0
        self.last_min_frontier_y = float('inf')
        self.last_max_frontier_y = 0
        
        self.max_viewpoint_x = 0
        self.min_viewpoint_x = float('inf')
        self.max_viewpoint_y = 0
        self.min_viewpoint_y = float('inf')
        self.last_max_viewpoint_x = 0
        self.last_min_viewpoint_x = float('inf')
        self.last_max_viewpoint_y = 0
        self.last_min_viewpoint_y = float('inf')
        
        self.node_type = [0, 1, 2, 3, 4] # node 的种类编号, 分别是frontier, viewpoint, robot自身, 其他robot, 和当前location 相连接的viewpoint 
        self.frontier_node_feature = []

        self.viewpoint_node_feature = {}

        self.frontier_dict = dict()
        self.viewpoint_dict =  dict()
        self.drone_pos_dict = dict()

        # 画图
        self.image = None
        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show()
    def discover_and_subscribe(self):
        # 获取当前可用的话题列表
        topic_list = self.get_topic_names_and_types()
        self.subscription_list = []
        # 正则表达式匹配 odometry 话题
        odom_pattern = re.compile(r'/state_ukf/odom_(\d+)')
        self.drone_num = 0 
        for topic, types in topic_list:
            odom_match = odom_pattern.match(topic)
            # 从话题列表中找到所有的无人机的odometry话题， uav_id 从实际的topic中提取
            if odom_match and 'nav_msgs/msg/Odometry' in types:
                uav_id = odom_match.group(1)
                subscription = self.create_subscription(
                    Odometry, 
                    topic, 
                    lambda msg, id=uav_id: self.get_uav_odom(msg, id), 
                    10
                )
                self.subscription_list.append(subscription)
                # {Fore.GREEN}frontier_dict: {self.frontier_dict}{Style.RESET_ALL}
                self.get_logger().info(f'{Fore.GREEN}fSubscribed to odometry topic: {topic} {Style.RESET_ALL}')
                self.drone_num += 1
        # 拓扑图相关参数
        self.neighbor_flag = [0 for i in range(self.drone_num)]
        self.robot_node_feature = [None for i in range(self.drone_num)]
        self.other_robot_node_feature = [[] for i in range(self.drone_num)]
        self.joint_robot_node_feature = [[] for i in range(self.drone_num)]
        self.robot_positions = np.array([[0, 0] for i in range(self.drone_num)])
        self.last_robot_positions = np.array([[0, 0] for i in range(self.drone_num)])
    
    # 更新地图
    def listener_callback(self, msg):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.map_data = np.array(msg.data).reshape((self.height, self.width))
        # self.get_logger().info(f'width: {self.width}, height: {self.height}, resolution: {self.resolution}')
        self.update_viewrange()
    def get_uav_odom(self, msg, uav_id):
        self.get_logger().info(f'UAV {uav_id} odom: {msg.pose.pose.position.x}, {msg.pose.pose.position.y}')
        ros_x = msg.pose.pose.position.x
        ros_y = msg.pose.pose.position.y
        map_x = int((-ros_x + self.map_origin[1]) / (self.resolution * self.downsample_factor))
        map_y = int((-ros_y + self.map_origin[0]) / (self.resolution * self.downsample_factor))
        self.uav_odoms[uav_id] = (map_x, map_y)


        
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
        try:
            for uav_id in self.uav_odoms:
                self.get_logger().info(f'self.uav_odoms[uav_id]: {self.uav_odoms[uav_id]}')
                rgb_map[self.uav_odoms[uav_id][0], self.uav_odoms[uav_id][1]] = [255, 0, 0]
        except:
            self.get_logger().info(f'{Fore.RED}No uav_odoms{Style.RESET_ALL}')
        # self.get_logger().info(f"{Fore.GREEN}frontier_dict: {self.frontier_dict}{Style.RESET_ALL}")
        # 在原图上绘制红色圆圈
        # rgb_map[bottom:top, left:right][circle_mask] = [255, 0, 0]
        
        if self.image is None:
            self.image = self.ax.imshow(rgb_map, origin='lower')
        else:
            self.image.set_data(rgb_map)
        # self.get_logger().info(f'rgb_map: {rgb_map}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def update_viewrange(self):  
        if self.map_data is None or len(self.uav_odoms) < self.drone_num:
            return
        # self.get_logger().info(f'self.uav_odoms: {self.uav_odoms}')
        # apply a spleep of 0.1 second in ros2, tell me how
        
        
        # self.get_logger().info(f'uav_odoms: {self.uav_odoms}')
        # 计算方框的边界
        # 当需要遍历所有 UAV 时
        for uav_id in self.uav_odoms:
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
            
            # 创建圆形mask
            # circle_mask = np.abs(dist_from_center - radius) < 0.5        
            circle_mask = dist_from_center  < 5//self.resolution     
            # 获取圆形边界上的点
            y_indices, x_indices = np.nonzero(circle_mask)

            # 在有效的点上进行循环
            for i in range(len(y_indices)):
                y_local = y_indices[i]
                x_local = x_indices[i]
                
                # 转换回全局坐标
                y_global = y_local + left 
                x_global = x_local + bottom
                # 这里似乎有问题
                if x_global >= self.map_size_height or y_global >= self.map_size_width:
                    continue
                # 在这里执行你想要的操作
                # self.get_logger().info(f'x: {x_global}, y: {y_global}')
                self.update_frontier(self.map_data, x_global, y_global)
        # self.get_logger().info(f'frontier_dict: {self.frontier_dict}')

        for uav_id in self.uav_odoms:
            self.update_viewpoints(self.uav_odoms[uav_id][0], self.uav_odoms[uav_id][1])    
        self.delete_view_point() # 只要一次就够
    
        self.occupancy_to_rgb(self.map_data)
        
        # self.get_logger().info("!!!!!!!!!!!!!!")
        
        # robot node feature
        # 根据环境制作当前的动态图, 核心代码
        
        for i, ego_uav_id in enumerate(self.uav_odoms.keys()):
            self.robot_node_feature[i] = [self.uav_odoms[ego_uav_id][0], self.uav_odoms[ego_uav_id][1], 0, self.node_type[2]]
            for each_uav_id in self.uav_odoms:
                if each_uav_id != ego_uav_id:
                    self.other_robot_node_feature[i].append([self.uav_odoms[each_uav_id][0], self.uav_odoms[each_uav_id][1], 0, self.node_type[3]])
            self.joint_robot_node_feature[i] = [self.uav_odoms[ego_uav_id][0], self.uav_odoms[ego_uav_id][1], 0, self.node_type[2]]
        # 安全检测，防止没有处理好 frontier 导致下面代码出错
        if not self.frontier_dict:
            return
        self.graph_list, joint_graph_data, indices_viewpoint, self.robot_positions[:, :2], no_viewpoint = self.node_feature_process(self.frontier_node_feature, self.viewpoint_node_feature, self.robot_node_feature, self.other_robot_node_feature, self.joint_robot_node_feature)
        if not self.viewpoint_node_feature:
            done = [True for i in range(self.drone_num)]
            self.graph_list, joint_graph_data, indices_viewpoint = self.reset()
        else: 
            self.graph_list, joint_graph_data, indices_viewpoint, self.robot_positions, no_viewpoint = self.node_feature_process(self.frontier_node_feature, self.viewpoint_node_feature, self.robot_node_feature, self.other_robot_node_feature, self.joint_robot_node_feature)
        if no_viewpoint:
            done = [True for i in range(self.drone_num)]
        self.get_logger().info(f'{Fore.GREEN}indices_viewpoint: {indices_viewpoint}{Style.RESET_ALL}')
    
    def update_frontier(self, data, x, y):
        
        # 防止frontier的检测超出地图的边界
        min_x = max(0, x - 1)
        min_y = max(0, y - 1)
        max_x = min(self.map_size_height - 1, x + 1)
        max_y = min(self.map_size_width - 1, y + 1)
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
            # self.get_logger().info(f'{Fore.GREEN}Have frontier!{Style.RESET_ALL}')
            for nx, ny in neighbors:
                # 如果neighbor中的点有一个是未知区域，那么在frontier这个字典中添加这个neighbor点
                if data[nx, ny] == -1:
                    # 边界点的特征[点的种类编号, 周围有多少个未知区域]
                    self.frontier_dict[(nx, ny)] = [self.node_type[0], utility]
                    self.max_frontier_x = max(nx, self.max_frontier_x)
                    self.min_frontier_x = min(nx, self.min_frontier_x)
                    self.max_frontier_y = max(ny, self.max_frontier_y)
                    self.min_frontier_y = min(ny, self.min_frontier_y)
                    self.frontier_node_feature.append([nx, ny, utility, self.node_type[0]])
                # 排除不属于边点
                elif (nx, ny) in self.frontier_dict:
                    self.frontier_node_feature.remove([nx, ny, self.frontier_dict[(nx, ny)][-1], self.node_type[0]])
                    del self.frontier_dict[(nx, ny)]
                    if self.frontier_dict:
                        if self.max_frontier_x == nx:
                            self.max_frontier_x = max([key[0] for key in self.frontier_dict.keys()])                            
                        if self.min_frontier_x == nx:
                            self.min_frontier_x = min([key[0] for key in self.frontier_dict.keys()])
                        if self.max_frontier_y == ny:
                            self.max_frontier_y = max([key[1] for key in self.frontier_dict.keys()])
                        if self.min_frontier_y == ny:
                            self.min_frontier_y = min([key[1] for key in self.frontier_dict.keys()])
                    else:
                        self.max_frontier_x = 0
                        self.min_frontier_x = 0
                        self.max_frontier_y = 0
                        self.min_frontier_y = 0
        else:
            # 如果neighbor中的点都是已经探索过了，那么在frontier这个字典中检查是否有这个neighbor点，有的话就删除
            for nx, ny in neighbors:
                if (nx, ny) in self.frontier_dict:
                    self.frontier_node_feature.remove([nx, ny, self.frontier_dict[(nx, ny)][-1], self.node_type[0]])
                    del self.frontier_dict[(nx, ny)]
                    if self.frontier_dict:
                        if self.max_frontier_x == nx:
                            self.max_frontier_x = max([key[0] for key in self.frontier_dict.keys()])                            
                        if self.min_frontier_x == nx:
                            self.min_frontier_x = min([key[0] for key in self.frontier_dict.keys()])
                        if self.max_frontier_y == ny:
                            self.max_frontier_y = max([key[1] for key in self.frontier_dict.keys()])
                        if self.min_frontier_y == ny:
                            self.min_frontier_y = min([key[1] for key in self.frontier_dict.keys()])
                    else:
                        self.max_frontier_x = 0
                        self.min_frontier_x = 0
                        self.max_frontier_y = 0
                        self.min_frontier_y = 0

    def ray_point(self, x0, y0, x1, y1):
        points = []
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
    
        if dx == 0:
            while y0 != y1:
                y0 += sy
                points.append((x0, y0))
            return points
        elif dy == 0:
            while x0 != x1:
                x0 += sx
                points.append((x0, y0))
            return points
        
        while True:
            if x0 == x1 and y0 == y1:
                break
            if x0 == x1:
                while y0 != y1:
                    y0 += sy
                    points.append((x0, y0))
                break
            if y0 == y1:
                while x0 != x1:
                    x0 += sx
                    points.append((x0, y0))
                break
            x0 += sx
            y0 += sy
            points.append((x0, y0))
        return points

    #  输入的是那个无人机的位置，就根据这个位置更新当前的viewpoint
    def update_viewpoints(self, x, y):
        # update view_points
        min_x = max(0, x-1)
        min_y = max(0, y-1)
        max_x = min(self.map_size_height, x+1)
        max_y = min(self.map_size_width, y+1)
        # ii, jj = np.mgrid[min_x:max_x+1, min_y:max_y+1]
        # candidate_view_points = np.stack([ii.ravel(), jj.ravel()], axis=-1)
        candidate_view_points = [(min_x, y), (max_x, y), (x, min_y), (x, max_y)]
        self.get_logger().info(f'candidate_view_points: {candidate_view_points}')
        # Find unknown grids
        for nx, ny in candidate_view_points:
            view_min_x = max(0, nx-self.viewrange + 1)
            view_min_y = max(0, ny-self.viewrange + 1)
            view_max_x = min(self.map_size_height, nx+self.viewrange)
            view_max_y = min(self.map_size_width, ny+self.viewrange)
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
                    frontier_coordinate_list.append((int(vx), int(vy)))
                    utility += 1

            if frontier_coordinate_list:
                for frontier_x, frontier_y in frontier_coordinate_list:
                    check_points = self.ray_point(frontier_x, frontier_y, nx, ny)
                    for point in check_points:
                        # self.get_logger().info(f'{Fore.GREEN}point: {point}{Style.RESET_ALL}')
                        if self.map_data[point[0], point[1]] == 100:
                            frontier_coordinate_list.remove((frontier_x, frontier_y))
                            utility -= 1
                            break

            if frontier_coordinate_list:
                self.max_viewpoint_x = max(nx, self.max_viewpoint_x)
                self.min_viewpoint_x = min(nx, self.min_viewpoint_x)
                self.max_viewpoint_y = max(ny, self.max_viewpoint_y)
                self.min_viewpoint_y = min(ny, self.min_viewpoint_y)
                self.viewpoint_dict[(nx, ny)] = [self.node_type[1], utility, frontier_coordinate_list]
                self.viewpoint_node_feature[(nx, ny)]=[nx, ny, utility, self.node_type[1]]

    def delete_view_point(self):
        delete_item= []
        for uav_id in self.uav_odoms:
            if (self.uav_odoms[uav_id][0],  self.uav_odoms[uav_id][1]) in self.viewpoint_dict:
                delete_item.append((self.uav_odoms[uav_id][0],  self.uav_odoms[uav_id][1]))
                
        self.delete_num = 0
        for (nx, ny), (node_type, utility, frontier_coordinate_list) in self.viewpoint_dict.items():
            if self.map_data[nx, ny] == 100 :
                delete_item.append((nx, ny))                
                continue
            view_min_x = max(0, nx-self.viewrange + 1)
            view_min_y = max(0, ny-self.viewrange + 1)
            view_max_x = min(self.map_size_height, nx+self.viewrange)
            view_max_y = min(self.map_size_width, ny+self.viewrange)
            ii, jj = np.mgrid[view_min_x:view_max_x, view_min_y:view_max_y]
            view_points_view_range = np.stack([ii.ravel(), jj.ravel()], axis=-1)
            # how many frontiers can be observed by this view point
            utility = 0
            exist_frontier = False
            for vx, vy in frontier_coordinate_list:
                unknown = False
                block = False
                # 如果是未知点
                if self.map_data[vx, vy] == -1:
                    unknown = True
                    # 如果这个未知点到viewpoint的连线上没有障碍物                    
                    check_points = self.ray_point(vx, vy, nx, ny)
                    for point in check_points:
                        if self.map_data[point[0], point[1]] == 100:
                            block = True
                            break
                if unknown and not block:
                    utility += 1
                    exist_frontier = True
            # 这里应该更新每个viewpoint的utility
            self.viewpoint_dict[(nx, ny)][1] = utility
            self.viewpoint_node_feature[(nx, ny)][2] = utility
            if not exist_frontier:
                self.delete_num += 1
                delete_item.append((nx, ny))

        for delete_coordinate in delete_item: 
            if (delete_coordinate[0], delete_coordinate[1]) not in self.viewpoint_node_feature:
                continue
            del self.viewpoint_node_feature[(delete_coordinate[0], delete_coordinate[1])]
            del self.viewpoint_dict[(delete_coordinate[0], delete_coordinate[1])]
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
        for uav_id in self.uav_odoms:
            min_x = max(0, self.uav_odoms[uav_id][0] -1)
            min_y = max(0,  self.uav_odoms[uav_id][1]-1)
            max_x = min(self.map_size_height,  self.uav_odoms[uav_id][0]+1)
            max_y = min(self.map_size_width,  self.uav_odoms[uav_id][1]+1)
            # ii, jj = np.mgrid[min_x:max_x+1, min_y:max_y+1]
            # candidate_view_points = np.stack([ii.ravel(), jj.ravel()], axis=-1)
            candidate_view_points = [(min_x,  self.uav_odoms[uav_id][1]), (max_x,  self.uav_odoms[uav_id][1]), ( self.uav_odoms[uav_id][0], min_y), ( self.uav_odoms[uav_id][0], max_y)]
            for (nx, ny) in candidate_view_points:
                if (nx, ny) in self.viewpoint_dict:
                    self.viewpoint_node_feature[(nx, ny)][-1] = self.node_type[4]
                    self.viewpoint_dict[(nx, ny)][0] = self.node_type[4]

       
    def node_feature_process(self, frontier_node_feature, viewpoint_node_feature, robot_node_feature, other_robot_node_feature, joint_robot_node_feature):
        no_viewpoint = False
        width = self.max_frontier_x - self.min_frontier_x + 1
        height = self.max_frontier_y - self.min_frontier_y + 1
        list_viewpoint_ndoe_feature = list(viewpoint_node_feature.values())

        # 将列表转换为 NumPy 数组
        # self.get_logger().info(f'{Fore.RED}frontier_node_feature: {frontier_node_feature}{Style.RESET_ALL}')
        frontier_array = np.array(frontier_node_feature, dtype=np.float16)
        viewpoint_array = np.array(list_viewpoint_ndoe_feature, dtype=np.float16)
        robot_array = np.array(robot_node_feature, dtype=np.float16)
        other_robot_array = np.array(other_robot_node_feature, dtype=np.float16)
        joint_robot_array = np.array(joint_robot_node_feature, dtype=np.float16)
        # 处理坐标
        # print("frontier_array", frontier_array)
        frontier_array[:, 0] = (frontier_array[:, 0] - self.min_frontier_x) / width 
        frontier_array[:, 1] = (frontier_array[:, 1] - self.min_frontier_y) / height
        try:
            viewpoint_array[:, 0] = (viewpoint_array[:, 0] - self.min_frontier_x) / width
            viewpoint_array[:, 1] = (viewpoint_array[:, 1] - self.min_frontier_y) / height
        except:
            print("viewpoint_array", viewpoint_array)
            if len(viewpoint_array) == 0:
                no_viewpoint = True
            else:
                viewpoint_array = np.expand_dims(viewpoint_array, axis=0)
                viewpoint_array[:, 0] = (viewpoint_array[:, 0] - self.min_frontier_x) / width
                viewpoint_array[:, 1] = (viewpoint_array[:, 1] - self.min_frontier_y) / height
        robot_array[:, 0] = (robot_array[:, 0] - self.min_frontier_x) / width
        robot_array[:, 1] = (robot_array[:, 1] - self.min_frontier_y) / height
        other_robot_array[:,:,0] = (other_robot_array[:,:,0] - self.min_frontier_x) / width
        other_robot_array[:,:,1] = (other_robot_array[:,:,1] - self.min_frontier_y) / height
        joint_robot_array[:, 0] = (joint_robot_array[:, 0] - self.min_frontier_x) / width
        joint_robot_array[:, 1] = (joint_robot_array[:, 1] - self.min_frontier_y) / height
        # 计算边
        # 首先是各个点的数量
        num_frontier_node = frontier_array.shape[0]
        num_viewpoint_node = viewpoint_array.shape[0]
        num_robot_node = 1
        num_other_robot_node = self.drone_num - 1
        num_joint_robot_node = self.drone_num
        # 然后是从一类点到一类点的indices
        indices_frontier = list(range(num_frontier_node))
        indices_viewpoint = list(range(num_frontier_node, num_frontier_node + num_viewpoint_node))
        indices_other_robot = list(range(num_frontier_node + num_viewpoint_node, num_frontier_node + num_viewpoint_node + num_other_robot_node))
        indices_robot = list(range(num_frontier_node + num_viewpoint_node + num_other_robot_node, num_frontier_node + num_viewpoint_node + num_other_robot_node + 1))
        # criti网络中要用到的joitn obs的indices
        indices_joint_robot = list(range(num_frontier_node + num_viewpoint_node, num_frontier_node + num_viewpoint_node + num_joint_robot_node))
        
        ###### 仅仅是evaluation的时候使用
        self.indices_frontier = indices_frontier
        self.indices_viewpoint = indices_viewpoint
        self.indices_other_robot = indices_other_robot
        self.indices_robot = indices_robot
        ########################################
        
        
        # 继而构建有向边, 首先是边界点到viewpoint点
        edge_index = []
        # 预处理frontier列表
        frontier_coord_dict = defaultdict(list)
        for indice_frontier, sublist in enumerate(frontier_node_feature):
            coords = (sublist[0], sublist[1])
            frontier_coord_dict[coords] = indice_frontier
        
        for indice_viewpoint in indices_viewpoint:
            # 获取viewpoint的坐标, 首先减去边界点的数量, 让索引对应, 然后获取x, y坐标
            view_point_x = list_viewpoint_ndoe_feature[indice_viewpoint - num_frontier_node][0]
            view_point_y = list_viewpoint_ndoe_feature[indice_viewpoint - num_frontier_node][1]
             # 获取这个viewpoint连接的frontier
            if (view_point_x, view_point_y) not in self.viewpoint_dict:
                continue
            for frontier in self.viewpoint_dict[(view_point_x, view_point_y)][2]:
                # find the indices of the frontier in the list called frontier_node_feature
                indice_frontier = frontier_coord_dict[(frontier[0], frontier[1])]
                if indice_frontier == []:
                    continue
                edge_index.append([indice_frontier, indice_viewpoint])
        edge_index_list= [copy.deepcopy(edge_index) for i in range(self.drone_num)]
        for i in range(self.drone_num):
            for indice_other_robot in indices_other_robot:
                for indice_viewpoint in indices_viewpoint:
                    edge_index_list[i].append([indice_viewpoint, indice_other_robot])
            for indice_robot in indices_robot:
                for indice_viewpoint in indices_viewpoint:
                    edge_index_list[i].append([indice_viewpoint, indice_robot])   
        
        # 构建 critic网络的有向边
        for i in range(self.drone_num):
            for indice_joint_robot in indices_joint_robot:
                for indice_viewpoint in indices_viewpoint:
                    edge_index.append([indice_viewpoint, indice_joint_robot])
        
        frontier_tensor = torch.tensor(frontier_array, dtype=torch.float16)
        viewpoint_tensor = torch.tensor(viewpoint_array, dtype=torch.float16)
        robot_tensor = torch.tensor(robot_array, dtype=torch.float16)
        joint_robot_tesnor = torch.tensor(joint_robot_array, dtype=torch.float16)

        joint_graph_data_x = torch.cat([frontier_tensor, viewpoint_tensor, joint_robot_tesnor])
        joint_graph_data_edge = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        joint_graph_data = Data(x=joint_graph_data_x, edge_index=joint_graph_data_edge)
        
        data_list = []
        for i in range(self.drone_num):
            other_robot_node_tensor = torch.tensor(other_robot_array[i], dtype=torch.float16)
            x= torch.cat([frontier_tensor, viewpoint_tensor, robot_tensor, other_robot_node_tensor])  
            edge_tensor = torch.tensor(edge_index_list[i], dtype=torch.long).t().contiguous()
            # 去除孤立点的版本(如果去除了,那么需要重新计算边的索引和点的索引, 太麻烦了,所以暂时不去除孤立点, 反正孤立点也不影响计算)
            # data_list.append(graph_process.remove_isolated_nodes(Data(x=x, edge_index=edge_tensor)))
            data_list.append(Data(x=x, edge_index=edge_tensor))

        return data_list, joint_graph_data, indices_viewpoint, robot_array[:, :2], no_viewpoint
        

def main(args=None):
    rclpy.init(args=args)
    node = occupancy_to_graph()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        