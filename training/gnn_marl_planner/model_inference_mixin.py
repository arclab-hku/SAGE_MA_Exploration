from __future__ import annotations

import os
import re
import sys
import math
import time
import copy
from types import SimpleNamespace
from importlib import import_module
import numpy as np
import torch
from collections import defaultdict
from itertools import islice
from colorama import Fore, Style
from gym import spaces
from torch_geometric.data import Data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from .onnx_shared_dynamic import SharedDynamicOnnxPolicy
from .onnx_full_dynamic import FullDynamicOnnxPolicy
from .uav_index import uav_key_from_index

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback handled at runtime
    linear_sum_assignment = None

class ModelInferenceMixin:
    def _build_shared_actor_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            nn_type='gnn',
            hidden_size=512,
            gain=0.01,
            use_orthogonal=True,
            use_policy_active_masks=True,
            use_naive_recurrent_policy=False,
            use_recurrent_policy=False,
            recurrent_N=1,
            use_ReLU=True,
            gnn_hidden_size=32,
            gnn_num_heads=4,
            gnn_concat_heads=True,
            gnn_layer_N=1,
            gnn_use_ReLU=True,
            embed_hidden_size=128,
            embed_layer_N=2,
            embed_use_ReLU=True,
            embed_add_self_loop=False,
            global_aggr_type='mean',
            num_embeddings=1000,
            embedding_size=64,
            use_feature_normalization=True,
            use_utility_head=False,
            use_utility_attention=False,
            add_dropout=False,
            dropout_prob=0.1,
            # Hungarian assignment parameters
            neighbor_penalty=1000.0,
            neighbor_distance_threshold=0.1,
            # Gaming layer parameters (required by HungarianActor)
            use_gaming_layer=True,
            gaming_hidden_dim=128,
            gaming_steps=3,
            context_dim=32,
            gaming_loss_coef=0.1,
            algorithm_name='happo',
            max_agents=8,
            min_agents=2,
        )

    def _init_pt_shared_model(self):
        if not os.path.isdir(self.deploy_root):
            raise FileNotFoundError(f"deploy_root does not exist: {self.deploy_root}")
        if self.deploy_root not in sys.path:
            sys.path.insert(0, self.deploy_root)

        actor_module = import_module("algorithms.actor_critic_hungarian")
        shared_actor_cls = getattr(actor_module, "HungarianActor")

        model_path = self.shared_actor_path.strip()
        if not model_path:
            model_path = os.path.join(
                self.deploy_root,
                "results",
                "shared_grpo_advance_20260216_grpo_hungarian",
                "shared_actor.pt",
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"shared_actor.pt not found: {model_path}")
        self._resolved_shared_actor_path = model_path

        device = torch.device(self.pt_model_device)
        model_args = self._build_shared_actor_args()
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        action_space = spaces.Discrete(self.pt_action_space_n)
        model = shared_actor_cls(model_args, observation_space, action_space, device=device)
        checkpoint = torch.load(model_path, map_location=device)
        if self.pt_strict_load:
            model.load_state_dict(checkpoint, strict=True)
            self.get_logger().info("Loaded shared_actor.pt with strict=True")
        else:
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            self.get_logger().warn(
                f"Loaded shared_actor.pt with strict=False, missing={len(missing)}, unexpected={len(unexpected)}"
            )
        model.eval()
        self.shared_actor = model
        self.get_logger().info(
            f"PT shared backend ready: {model_path}, device={self.pt_model_device}, deterministic={self.pt_deterministic}"
        )

    def _init_onnx_shared_dynamic(self):
        if self.shared_actor is None:
            raise RuntimeError("shared_actor must be initialized before onnx_shared_dynamic")

        onnx_path = self.onnx_shared_path.strip()
        if not onnx_path:
            base_model_path = self._resolved_shared_actor_path
            if not base_model_path:
                base_model_path = os.path.join(
                    self.deploy_root,
                    "results",
                    "shared_grpo_advance_20260216_grpo_hungarian",
                    "shared_actor.pt",
                )
            onnx_path = os.path.join(
                os.path.dirname(base_model_path),
                "shared_actor_policy_head_dynamic.onnx",
            )

        self.shared_dynamic_onnx_policy = SharedDynamicOnnxPolicy(
            shared_actor=self.shared_actor,
            onnx_model_path=onnx_path,
            force_export=self.onnx_shared_force_export,
        )
        self.get_logger().info(
            f"onnx_shared_dynamic backend ready (PT graph encoder + ONNX dynamic policy head): "
            f"{onnx_path}, force_export={self.onnx_shared_force_export}"
        )

    def _init_onnx_full_dynamic(self):
        if self.shared_actor is None:
            raise RuntimeError("shared_actor must be initialized before onnx_full_dynamic")

        onnx_path = self.onnx_full_path.strip()
        if not onnx_path:
            base_model_path = self._resolved_shared_actor_path
            if not base_model_path:
                base_model_path = os.path.join(
                    self.deploy_root,
                    "results",
                    "shared_grpo_advance_20260216_grpo_hungarian",
                    "shared_actor.pt",
                )
            onnx_path = os.path.join(
                os.path.dirname(base_model_path),
                "shared_actor_full_dynamic.onnx",
            )

        self.full_dynamic_onnx_policy = FullDynamicOnnxPolicy(
            shared_actor=self.shared_actor,
            onnx_model_path=onnx_path,
            force_export=self.onnx_full_force_export,
        )
        self.get_logger().info(
            f"onnx_full_dynamic backend ready (ONNX graph encoder + ONNX dynamic policy head): "
            f"{onnx_path}, force_export={self.onnx_full_force_export}"
        )

    def init_model(self):
        self.actor_list = []
        self.shared_actor = None
        self.shared_dynamic_onnx_policy = None
        self.full_dynamic_onnx_policy = None
        if self.inference_backend == "pt_shared":
            self._init_pt_shared_model()
            return
        if self.inference_backend == "onnx_shared_dynamic":
            self._init_pt_shared_model()
            self._init_onnx_shared_dynamic()
            return
        if self.inference_backend == "onnx_full_dynamic":
            self._init_pt_shared_model()
            self._init_onnx_full_dynamic()
            return
        raise ValueError(
            f"Unknown inference_backend={self.inference_backend}, "
            "expected pt_shared or onnx_shared_dynamic or onnx_full_dynamic"
        )

    def _build_index_batches(self):
        indices_viewpoint_batch = []
        all_indices_viewpoint_batch = []
        full_lists = getattr(self, "all_indices_viewpoint", [])
        for i in range(self.drone_num):
            full_idx = []
            if i < len(full_lists) and full_lists[i]:
                full_idx = [int(v) for v in full_lists[i]]
            elif i < len(self.matching_indices) and self.matching_indices[i]:
                full_idx = [int(v) for v in self.matching_indices[i]]

            neighbor_idx = []
            if i < len(self.matching_indices) and self.matching_indices[i]:
                neighbor_idx = [int(v) for v in self.matching_indices[i]]
            elif full_idx:
                neighbor_idx = list(full_idx)

            if not full_idx:
                full_idx = [0]
            if not neighbor_idx:
                neighbor_idx = list(full_idx)
            elif full_idx:
                neighbor_set = set(neighbor_idx)
                ordered_neighbor_idx = [v for v in full_idx if v in neighbor_set]
                if ordered_neighbor_idx:
                    neighbor_idx = ordered_neighbor_idx

            # Align with training:
            # - indices_viewpoint_batch: actor input index list (matching_indices)
            # - all_indices_viewpoint_batch: full viewpoint index list
            # gatherNodeFeats() in the trained model uses the first list to gather
            # candidate node features, so swapping these changes the action semantics.
            indices_viewpoint_batch.append(neighbor_idx)
            all_indices_viewpoint_batch.append(full_idx)
        return indices_viewpoint_batch, all_indices_viewpoint_batch

    def _coordinate_joint_actions(
        self,
        mixed_probs: np.ndarray,
        indices_viewpoint_batch,
        all_indices_viewpoint_batch,
    ) -> np.ndarray:
        batch = int(mixed_probs.shape[0]) if mixed_probs is not None else 0
        fallback = np.zeros((batch, 1), dtype=np.int64)
        for i in range(batch):
            row = mixed_probs[i]
            fallback[i, 0] = int(np.argmax(row)) if row.size > 0 else 0

        if batch <= 1 or linear_sum_assignment is None:
            return fallback

        reference_full = None
        for full in all_indices_viewpoint_batch:
            if full:
                reference_full = [int(v) for v in full]
                break
        if not reference_full:
            return fallback

        for full in all_indices_viewpoint_batch:
            if full and [int(v) for v in full] != reference_full:
                self.get_logger().warn(
                    "Hungarian coordination skipped: per-agent all_indices_viewpoint ordering mismatch"
                )
                return fallback

        n_agents = batch
        n_viewpoints = len(reference_full)
        if n_viewpoints == 0:
            return fallback

        neighbor_mask = np.zeros((n_agents, n_viewpoints), dtype=bool)
        for i in range(n_agents):
            full = [int(v) for v in (all_indices_viewpoint_batch[i] if i < len(all_indices_viewpoint_batch) else [])]
            neigh = [int(v) for v in (indices_viewpoint_batch[i] if i < len(indices_viewpoint_batch) else [])]
            if not full:
                continue
            neigh_set = set(neigh)
            for j, vp in enumerate(full):
                if vp in neigh_set:
                    neighbor_mask[i, j] = True

        neighbor_penalty = float(getattr(self, "neighbor_penalty", 1000.0))
        # Keep non-neighbor assignments available as a last resort for Hungarian,
        # but make them decisively worse than any valid local neighbor action.
        steal_penalty = max(neighbor_penalty * 0.1, 50.0)
        non_neighbor_penalty = max(neighbor_penalty, 1000.0)
        cost_matrix = np.zeros((n_agents, n_viewpoints), dtype=np.float64)

        vp_owners = {}
        for j in range(n_viewpoints):
            owners = [i for i in range(n_agents) if neighbor_mask[i, j]]
            if owners:
                vp_owners[j] = owners

        for i in range(n_agents):
            my_neighbor_vps = np.where(neighbor_mask[i])[0].tolist()
            valid_k = len(indices_viewpoint_batch[i]) if i < len(indices_viewpoint_batch) else 0
            for j in range(n_viewpoints):
                is_my_neighbor = neighbor_mask[i, j]
                if is_my_neighbor and j in my_neighbor_vps:
                    neighbor_idx = my_neighbor_vps.index(j)
                    if neighbor_idx < valid_k:
                        prob = float(mixed_probs[i, neighbor_idx])
                        prob = max(prob, 1e-8)
                        cost_matrix[i, j] = -math.log(prob)
                    else:
                        cost_matrix[i, j] = 0.0
                else:
                    cost_matrix[i, j] = non_neighbor_penalty
                    if j in vp_owners and any(owner != i for owner in vp_owners[j]):
                        cost_matrix[i, j] += steal_penalty

        if n_viewpoints < n_agents:
            expanded_cost = np.tile(cost_matrix, (1, n_agents))
            row_ind, col_ind = linear_sum_assignment(expanded_cost)
            actions_list = (col_ind % n_viewpoints).tolist()
        else:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            actions_list = col_ind.tolist()

        actions = np.zeros((n_agents, 1), dtype=np.int64)
        for i in range(n_agents):
            action = int(actions_list[i]) if i < len(actions_list) else int(fallback[i, 0])
            actions[i, 0] = max(0, min(action, n_viewpoints - 1))
        return actions

    def discover_and_subscribe(self):
        # 获取当前可用的话题列表（启动期容忍 bridge/odom 发现延迟）
        self.subscription_list = []
        self.publisher_list = []
        odom_pattern = re.compile(r'/quad_(\d+)/lidar_slam/odom')
        self.drone_num = 0

        discovered = []
        topic_list = []
        deadline = time.monotonic() + self.odom_discovery_timeout_s
        attempt = 0
        while True:
            topic_list = self.get_topic_names_and_types()
            discovered = []
            for topic, types in topic_list:
                odom_match = odom_pattern.match(topic)
                if odom_match and 'nav_msgs/msg/Odometry' in types:
                    discovered.append((int(odom_match.group(1)), topic))
            if discovered or time.monotonic() >= deadline:
                break
            attempt += 1
            if attempt == 1 or (attempt % 4 == 0):
                self.get_logger().warn(
                    f"waiting odom topics from bridge... attempt={attempt}, "
                    f"timeout={self.odom_discovery_timeout_s:.1f}s"
                )
            time.sleep(self.odom_discovery_retry_s)

        discovered.sort(key=lambda x: x[0])
        self.get_logger().info(
            f"{Fore.GREEN}odom discovery done: found={len(discovered)} / topics={len(topic_list)}{Style.RESET_ALL}"
        )
        for uav_idx, topic in discovered:
            uav_id = str(uav_idx)
            subscription = self.create_subscription(
                Odometry,
                topic,
                lambda msg, id=uav_id: self.get_uav_odom(msg, id),
                10
            )
            self.subscription_list.append(subscription)
            self.drone_num += 1
            topic_name = f'move_base_simple/goal_{self.drone_num}'
            publisher = self.create_publisher(PoseStamped, topic_name, 10)
            self.publisher_list.append(publisher)

        if self.drone_num == 0:
            self.get_logger().error(
                "No odom topics discovered; planner will stay idle. "
                "Check ros1_bridge and /quad_*/lidar_slam/odom mapping."
            )
        # uav 的动作序列的初始化
        self.uav_action_list = [[] for i in range(self.drone_num)]
        # 拓扑图相关参数
        self.neighbor_flag = [0 for i in range(self.drone_num)]
        self.robot_node_feature = [None for i in range(self.drone_num)]
        self.other_robot_node_feature = [[] for i in range(self.drone_num)]
        self.joint_robot_node_feature = [[] for i in range(self.drone_num)]
        self.robot_positions = np.array([[0, 0] for i in range(self.drone_num)])
        self.last_robot_positions = np.array([[0, 0] for i in range(self.drone_num)])
        # 初始化动作
        self.action_list = [0 for i in range(self.drone_num)]
        # 初始化模型
        self.init_model()
        # 初始化到达标志
        self.arrive_flag = [True for i in range(self.drone_num)]        

    def node_feature_process(self, frontier_node_feature, viewpoint_node_feature, robot_node_feature, other_robot_node_feature, joint_robot_node_feature):
        width = self.max_frontier_x - self.min_frontier_x + 1
        height = self.max_frontier_y - self.min_frontier_y + 1

        # 提取每个字典的值并转换为 NumPy 数组
        list_viewpoint_ndoe_feature = [[d[k] for k in sorted(d.keys())] for d in viewpoint_node_feature]
        # 将列表转换为 NumPy 数组
        frontier_array = np.array(frontier_node_feature, dtype=np.float16)
        # dim(drone_num, num_viewpoint, 6)
        viewpoint_array = np.array(list_viewpoint_ndoe_feature, dtype=np.float16)
        self.get_logger().debug(f'viewpoint_array shape: {viewpoint_array.shape}')
        robot_array = np.array(robot_node_feature, dtype=np.float16)
        other_robot_array = np.array(other_robot_node_feature, dtype=np.float16)
        joint_robot_array = np.array(joint_robot_node_feature, dtype=np.float16)
        # 处理坐标
        frontier_array[:, 0] = (frontier_array[:, 0] - self.min_frontier_x) / width 
        frontier_array[:, 1] = (frontier_array[:, 1] - self.min_frontier_y) / height
        no_viewpoint = False
        try: 
            # 当viewpoint_array 为一个三维的array的时候， 中间记得把多的维度补上
            viewpoint_array[:, :, 0] = (viewpoint_array[:,:, 0] - self.min_frontier_x) / width
            viewpoint_array[:, :, 1] = (viewpoint_array[:,:, 1] - self.min_frontier_y) / height
        except:
            print("!!!!!!!!!!viewpoint_array", viewpoint_array)
            if len(viewpoint_array) == 0:
                no_viewpoint = True
            else:
                viewpoint_array = np.expand_dims(viewpoint_array, axis=0)
                viewpoint_array[:, :, 0] = (viewpoint_array[:, :, 0] - self.min_frontier_x) / width
                viewpoint_array[:, :, 1] = (viewpoint_array[:, :, 1] - self.min_frontier_y) / height
        robot_array[:, 0] = (robot_array[:, 0] - self.min_frontier_x) / width
        robot_array[:, 1] = (robot_array[:, 1] - self.min_frontier_y) / height
        other_robot_array[:,:,0] = (other_robot_array[:,:,0] - self.min_frontier_x) / width
        other_robot_array[:,:,1] = (other_robot_array[:,:,1] - self.min_frontier_y) / height
        joint_robot_array[:, 0] = (joint_robot_array[:, 0] - self.min_frontier_x) / width
        joint_robot_array[:, 1] = (joint_robot_array[:, 1] - self.min_frontier_y) / height
        # 计算边
        # 首先是各个点的数量
        num_frontier_node = frontier_array.shape[0]
        self.frontier_num = num_frontier_node
        num_viewpoint_node = viewpoint_array.shape[1]
        num_robot_node = 1
        num_other_robot_node = self.drone_num - 1
        num_joint_robot_node = self.drone_num
        # 然后是从一类点到一类点的indices
        indices_frontier = list(range(num_frontier_node))
        # neighbor_indices_viewpoint = [[] for i in range(self.drone_num)]
        # print("indices_frontier is", indices_frontier)
        # self.get_logger().info(f'{Fore.GREEN}before adding self.matching_indices: {self.matching_indices}{Style.RESET_ALL}')
        raw_all_indices = getattr(self, "all_indices_viewpoint", [[] for _ in range(self.drone_num)])
        if len(raw_all_indices) < self.drone_num:
            raw_all_indices = list(raw_all_indices) + [[] for _ in range(self.drone_num - len(raw_all_indices))]

        for i in range(self.drone_num):
            if self.matching_indices[i]:
                self.matching_indices[i] = [x + num_frontier_node for x in self.matching_indices[i]]
            if raw_all_indices[i]:
                raw_all_indices[i] = [x + num_frontier_node for x in raw_all_indices[i]]

        indices_viewpoint = list(range(num_frontier_node, num_frontier_node + num_viewpoint_node)) 
        self.all_indices_viewpoint = [
            list(raw_all_indices[i]) if raw_all_indices[i] else list(indices_viewpoint)
            for i in range(self.drone_num)
        ]
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

            view_point_x = list_viewpoint_ndoe_feature[0][indice_viewpoint - num_frontier_node][0]
            view_point_y = list_viewpoint_ndoe_feature[0][indice_viewpoint - num_frontier_node][1]
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
        
        mean_viewpoint_array = np.mean(viewpoint_array, axis=0)
        joint_viewpoint_tensor = torch.tensor(mean_viewpoint_array, dtype=torch.float16)
        joint_robot_tesnor = torch.tensor(joint_robot_array, dtype=torch.float16)
        joint_graph_data_x = torch.cat([frontier_tensor, joint_viewpoint_tensor, joint_robot_tesnor])
        joint_graph_data_edge = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        joint_graph_data = Data(x=joint_graph_data_x, edge_index=joint_graph_data_edge)
         
        data_list = []
        for i in range(self.drone_num):
            other_robot_node_tensor = torch.tensor(other_robot_array[i], dtype=torch.float16)
            robot_tensor = torch.tensor(robot_array[i], dtype=torch.float16).unsqueeze(0)
            viewpoint_tensor = torch.tensor(viewpoint_array[i], dtype=torch.float16)
            x= torch.cat([frontier_tensor, viewpoint_tensor, robot_tensor, other_robot_node_tensor])  
            edge_tensor = torch.tensor(edge_index_list[i], dtype=torch.long).t().contiguous()
            # 去除孤立点的版本(如果去除了,那么需要重新计算边的索引和点的索引, 太麻烦了,所以暂时不去除孤立点, 反正孤立点也不影响计算)
            # data_list.append(graph_process.remove_isolated_nodes(Data(x=x, edge_index=edge_tensor)))
            data_list.append(Data(x=x, edge_index=edge_tensor))
        self.get_logger().debug(f'self.matching_indices: {self.matching_indices}')
        for i, list_i in enumerate(self.matching_indices):
            self.is_neighbor[i] = True
            if not list_i:
                self.is_neighbor[i] = False
                self.matching_indices[i] = list(self.all_indices_viewpoint[i])
        return data_list, joint_graph_data,  self.matching_indices, robot_array[:, :2] , no_viewpoint    
