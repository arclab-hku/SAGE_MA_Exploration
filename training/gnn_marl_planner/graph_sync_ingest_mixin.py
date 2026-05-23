from __future__ import annotations

import json
import math
import time
import numpy as np
from pathlib import Path
from .uav_index import uav_key_from_index

_DIAG_PATH = Path("/tmp/gnn_marl_graph_diag.json")

class GraphSyncIngestMixin:
    def _current_graph_snapshot(self):
        graph_nodes = (
            self._graph_nodes_committed
            if self._graph_snapshot_committed_once
            else self._graph_nodes_merged
        )
        graph_edges = (
            self._graph_edges_committed
            if self._graph_snapshot_committed_once
            else self._graph_edges_merged
        )
        return graph_nodes, graph_edges

    def _graph_node_counts(self):
        graph_nodes, graph_edges = self._current_graph_snapshot()
        frontier = sum(1 for n in graph_nodes.values() if int(n["node_type"]) == 0)
        viewpoint = sum(1 for n in graph_nodes.values() if int(n["node_type"]) == 1)
        robot = sum(1 for n in graph_nodes.values() if int(n["node_type"]) == 2)
        return {
            "nodes": int(len(graph_nodes)),
            "edges": int(len(graph_edges)),
            "frontier": int(frontier),
            "viewpoint": int(viewpoint),
            "robot": int(robot),
        }

    def _commit_live_graph_snapshot(self, reason: str) -> bool:
        if self.map_input_mode != "graph_sync":
            return False
        if not self._graph_nodes_merged and not self._graph_edges_merged and self._graph_snapshot_committed_once:
            return False
        if (
            self._graph_snapshot_committed_once
            and self._graph_committed_topology_updates == self._graph_topology_updates
        ):
            return False

        self._graph_nodes_committed = {
            node_id: dict(node) for node_id, node in self._graph_nodes_merged.items()
        }
        self._graph_edges_committed = [dict(edge) for edge in self._graph_edges_merged]
        self._graph_committed_topology_updates = int(self._graph_topology_updates)
        self._graph_snapshot_committed_once = True
        self._last_snapshot_commit_ts = time.monotonic()
        self.get_logger().info(
            f"graph_snapshot commit: reason={reason}, nodes={len(self._graph_nodes_committed)}, "
            f"edges={len(self._graph_edges_committed)}, live_update_id={self._graph_topology_updates}"
        )
        if hasattr(self, "_append_event_log"):
            self._append_event_log(
                "graph_snapshot_commit",
                reason=str(reason),
                nodes=int(len(self._graph_nodes_committed)),
                edges=int(len(self._graph_edges_committed)),
                live_update_id=int(self._graph_topology_updates),
            )
        return True

    def _rebuild_frontiers_from_graph(self):
        """Authoritative topology rebuild from merged GraphDelta snapshots."""
        self.frontier_dict = {}
        self.frontier_node_feature = []
        self.viewpoint_dict = {}
        self.viewpoint_node_feature = [{} for _ in range(self.drone_num)]
        self._reset_frontier_bounds()
        self.max_viewpoint_x = 0
        self.min_viewpoint_x = float('inf')
        self.max_viewpoint_y = 0
        self.min_viewpoint_y = float('inf')

        # Asymmetric commit semantics:
        # - Frontier nodes: always from LIVE merged graph (aggressive refresh)
        # - Viewpoint nodes: from committed snapshot (action-horizon stability)
        # - Edges: from LIVE merged graph (must match live frontier IDs,
        #   otherwise committed edges reference stale frontier IDs that
        #   no longer exist in the live frontier set, causing viewpoints
        #   to lose all frontier support)
        frontier_source_nodes = self._graph_nodes_merged
        vp_source_nodes = (
            self._graph_nodes_committed
            if self._graph_snapshot_committed_once
            else self._graph_nodes_merged
        )
        graph_edges = self._graph_edges_merged

        frontier_by_id = {}
        viewpoint_by_id = {}
        frontier_occ_conflict = 0
        viewpoint_occ_conflict = 0
        consumed_vp_filtered = 0
        consumed_filtered_coords = []
        raw_viewpoint_coords = []

        # Frontier nodes from live graph — disappear as soon as ROS1 clears them
        for node_id, node in frontier_source_nodes.items():
            if int(node["node_type"]) != 0:
                continue
            x = int(node["x"])
            y = int(node["y"])
            if not self._in_map(x, y):
                continue
            if int(self.downsampled_map_data[x, y]) == 100:
                frontier_occ_conflict += 1
            frontier_by_id[node_id] = (x, y)
            self.frontier_dict[(x, y)] = [self.node_type[0], 1]
            self.max_frontier_x = max(x, self.max_frontier_x)
            self.min_frontier_x = min(x, self.min_frontier_x)
            self.max_frontier_y = max(y, self.max_frontier_y)
            self.min_frontier_y = min(y, self.min_frontier_y)

        # Viewpoint nodes from committed snapshot — stable for action continuity
        for node_id, node in vp_source_nodes.items():
            if int(node["node_type"]) != 1:
                continue
            x = int(node["x"])
            y = int(node["y"])
            if not self._in_map(x, y):
                continue
            if (int(x), int(y)) in self._consumed_viewpoints_global:
                consumed_vp_filtered += 1
                if len(consumed_filtered_coords) < 16:
                    consumed_filtered_coords.append([int(x), int(y)])
                continue
            if int(self.downsampled_map_data[x, y]) == 100:
                viewpoint_occ_conflict += 1
            viewpoint_by_id[node_id] = (x, y)
            if len(raw_viewpoint_coords) < 16:
                raw_viewpoint_coords.append([int(x), int(y)])

        self.frontier_node_feature = [
            [nx, ny, int(val[-1]), 0, 0, self.node_type[0]]
            for (nx, ny), val in self.frontier_dict.items()
        ]

        vp_frontier_ids = {}
        edge_type0_count = 0
        edge_src_miss = 0
        edge_dst_miss = 0
        for edge in graph_edges:
            if int(edge["edge_type"]) != 0:
                continue
            edge_type0_count += 1
            src_id = edge["src_id"]
            dst_id = edge["dst_id"]
            if src_id not in frontier_by_id:
                edge_src_miss += 1
                continue
            if dst_id not in viewpoint_by_id:
                edge_dst_miss += 1
                continue
            vp_frontier_ids.setdefault(dst_id, []).append(src_id)

        diag = {
            "ts": time.time(),
            "live_nodes": len(self._graph_nodes_merged),
            "live_edges": len(self._graph_edges_merged),
            "frontier_src": "live",
            "vp_src": "committed" if self._graph_snapshot_committed_once else "live",
            "committed_vp_nodes": len(vp_source_nodes),
            "committed_edges": len(graph_edges),
            "frontiers": len(frontier_by_id),
            "viewpoints_raw": len(viewpoint_by_id),
            "frontier_occ_conflict": frontier_occ_conflict,
            "viewpoint_occ_conflict": viewpoint_occ_conflict,
            "consumed_vp_filtered": consumed_vp_filtered,
            "edges_type0": edge_type0_count,
            "edge_src_miss": edge_src_miss,
            "edge_dst_miss": edge_dst_miss,
            "vp_with_frontiers": len(vp_frontier_ids),
            "consumed_viewpoints_total": len(self._consumed_viewpoints_global),
        }
        self.get_logger().info(
            f"rebuild_graph: {json.dumps(diag)}"
        )
        try:
            _DIAG_PATH.write_text(json.dumps(diag, indent=2))
        except Exception:
            pass

        vp_candidates = []
        for vp_id, frontier_ids in vp_frontier_ids.items():
            if not frontier_ids:
                continue
            vx, vy = viewpoint_by_id[vp_id]
            seen = set()
            frontier_coords = []
            for fid in frontier_ids:
                if fid in seen:
                    continue
                seen.add(fid)
                frontier_coords.append(frontier_by_id[fid])
            if not frontier_coords:
                continue

            utility = len(frontier_coords)
            if self.uav_odoms:
                min_dist_sq = min(
                    (vx - int(odom[0])) * (vx - int(odom[0])) + (vy - int(odom[1])) * (vy - int(odom[1]))
                    for odom in self.uav_odoms.values()
                )
            else:
                min_dist_sq = 0
            vp_candidates.append((vp_id, vx, vy, frontier_coords, utility, min_dist_sq))

        if self.max_policy_viewpoints > 0 and len(vp_candidates) > self.max_policy_viewpoints:
            vp_candidates.sort(key=lambda item: (-int(item[4]), int(item[5])))
            vp_candidates = vp_candidates[: self.max_policy_viewpoints]
            self.get_logger().debug(
                f"prune viewpoints for policy: keep={len(vp_candidates)} limit={self.max_policy_viewpoints}"
            )

        kept_viewpoints = []

        for vp_id, vx, vy, frontier_coords, utility, _min_dist_sq in vp_candidates:
            self.viewpoint_dict[(vx, vy)] = [self.node_type[1], utility, frontier_coords]
            self.max_viewpoint_x = max(vx, self.max_viewpoint_x)
            self.min_viewpoint_x = min(vx, self.min_viewpoint_x)
            self.max_viewpoint_y = max(vy, self.max_viewpoint_y)
            self.min_viewpoint_y = min(vy, self.min_viewpoint_y)

            for drone_id in range(self.drone_num):
                drone_key = uav_key_from_index(self.uav_odoms, drone_id)
                if drone_key is None:
                    continue
                eff_r, eff_c = self._effective_pos_for_graph(drone_id, drone_key)
                to_drone_distance_x = vx - eff_r
                to_drone_distance_y = vy - eff_c
                to_drone_distance_x = math.copysign(
                    min(abs(to_drone_distance_x), self.classification_range), to_drone_distance_x
                )
                to_drone_distance_y = math.copysign(
                    min(abs(to_drone_distance_y), self.classification_range), to_drone_distance_y
                )
                self.viewpoint_node_feature[drone_id][(vx, vy)] = [
                    vx, vy, utility, to_drone_distance_x, to_drone_distance_y, self.node_type[1]
                ]
            if len(kept_viewpoints) < 16:
                kept_viewpoints.append({
                    "coord": [int(vx), int(vy)],
                    "utility": int(utility),
                    "frontier_count": int(len(frontier_coords)),
                })

        if (
            hasattr(self, "_append_event_log")
            and (
                int(consumed_vp_filtered) > 0
                or int(diag["vp_with_frontiers"]) == 0
                or int(len(vp_candidates)) <= 3
            )
        ):
            self._append_event_log(
                "rebuild_viewpoint_trace",
                frontier_count=int(len(frontier_by_id)),
                viewpoints_raw=int(len(viewpoint_by_id)),
                vp_with_frontiers=int(diag["vp_with_frontiers"]),
                consumed_vp_filtered=int(consumed_vp_filtered),
                consumed_viewpoints_total=int(len(self._consumed_viewpoints_global)),
                edge_src_miss=int(edge_src_miss),
                edge_dst_miss=int(edge_dst_miss),
                raw_viewpoint_coords=raw_viewpoint_coords,
                consumed_filtered_coords=consumed_filtered_coords,
                kept_viewpoints=kept_viewpoints,
            )

    def _apply_layer2_cell(self, row: int, col: int, state: int, obs_count: int, remote_lamport: int, remote_source: int):
        self.downsampled_map_data[row, col] = np.int8(state)
        self._layer2_obs_count[row, col] = np.uint8(obs_count)
        self._layer2_ver_lamport[row, col] = np.uint32(remote_lamport)
        self._layer2_ver_source[row, col] = np.int32(remote_source)

    def _on_layer2_delta(self, msg):
        if self.map_input_mode != "graph_sync":
            return
        if self._layer2_ver_lamport is None:
            return

        changed = 0
        min_override_obs = 3
        for cell in msg.changed_cells:
            cid = int(cell.cell_id)
            row = cid // self.map_size_width
            col = cid % self.map_size_width
            if row < 0 or col < 0 or row >= self.map_size_height or col >= self.map_size_width:
                continue

            local_ver = (
                int(self._layer2_ver_lamport[row, col]),
                int(self._layer2_ver_source[row, col]),
            )
            remote_ver = (int(cell.version_lamport), int(cell.version_source))

            local_state = int(self.downsampled_map_data[row, col])
            remote_state = int(cell.state)
            remote_obs = int(cell.obs_count)

            should_apply = False
            if remote_ver > local_ver:
                # Safety guard: low-confidence FREE cannot override local OCCUPIED.
                if (
                    local_state == 100
                    and remote_state != 100
                    and remote_obs < min_override_obs
                ):
                    should_apply = False
                else:
                    should_apply = True
            elif remote_ver == local_ver:
                remote_pri = self._state_priority(remote_state)
                local_pri = self._state_priority(local_state)
                if remote_pri > local_pri and remote_obs >= min_override_obs:
                    should_apply = True
                elif remote_pri == local_pri and remote_obs > int(self._layer2_obs_count[row, col]):
                    should_apply = True

            if should_apply:
                self._apply_layer2_cell(
                    row=row,
                    col=col,
                    state=remote_state,
                    obs_count=remote_obs,
                    remote_lamport=remote_ver[0],
                    remote_source=remote_ver[1],
                )
                changed += 1

        if changed > 0:
            self.map_update_flag = True
            self._last_layer2_delta_ts = time.monotonic()
            self.get_logger().debug(
                f"layer2_delta merged: source={msg.source_uav}, changed={changed}, seq={msg.seq}"
            )

    def _on_graph_delta(self, msg):
        if self.map_input_mode != "graph_sync":
            return
        source = int(msg.source_uav)
        incoming_nodes = {}
        for node in msg.nodes:
            if bool(node.deleted):
                continue
            node_id = node.node_id if node.node_id else f"n_{int(node.node_type)}_{int(node.x)}_{int(node.y)}"
            incoming_nodes[node_id] = {
                "node_id": node_id,
                "node_type": int(node.node_type),
                "x": int(node.x),
                "y": int(node.y),
                "ver": (int(node.version_lamport), int(node.version_source)),
            }

        incoming_edges = []
        for edge in msg.edges:
            incoming_edges.append(
                {
                    "src_id": edge.src_id,
                    "dst_id": edge.dst_id,
                    "edge_type": int(edge.edge_type),
                    "blocked": bool(edge.blocked),
                    "ver": (int(edge.version_lamport), int(edge.version_source)),
                }
            )

        self._graph_source_snapshots[source] = {
            "lamport": int(msg.lamport_clock),
            "nodes": incoming_nodes,
            "edges": incoming_edges,
        }

        merged_nodes = {}
        for snap in self._graph_source_snapshots.values():
            for node_id, node in snap["nodes"].items():
                exist = merged_nodes.get(node_id)
                if exist is None or node["ver"] > exist["ver"]:
                    merged_nodes[node_id] = node

        merged_edges = {}
        for snap in self._graph_source_snapshots.values():
            for edge in snap["edges"]:
                key = (edge["src_id"], edge["dst_id"], edge["edge_type"])
                exist = merged_edges.get(key)
                if exist is None or edge["ver"] > exist["ver"]:
                    merged_edges[key] = edge

        self._graph_nodes_merged = merged_nodes
        self._graph_edges_merged = list(merged_edges.values())
        self._graph_delta_recv_count += 1
        if msg.nodes or msg.edges:
            self._graph_topology_updates += 1
            self.map_update_flag = True
            self._last_graph_delta_ts = time.monotonic()
        if self._graph_delta_recv_count % 50 == 1:
            active_frontiers = sum(
                1 for n in self._graph_nodes_merged.values() if int(n["node_type"]) == 0
            )
            active_viewpoints = sum(
                1 for n in self._graph_nodes_merged.values() if int(n["node_type"]) == 1
            )
            self.get_logger().info(
                f"graph_delta recv: source={msg.source_uav}, lamport={msg.lamport_clock}, "
                f"nodes={len(msg.nodes)}, edges={len(msg.edges)}, merged_frontiers={active_frontiers}, "
                f"merged_viewpoints={active_viewpoints}"
            )

    def _on_state_digest(self, msg):
        if self.map_input_mode != "graph_sync":
            return
        self._state_digest_recv_count += 1
        if self._state_digest_recv_count % 100 == 1:
            self.get_logger().debug(
                f"state_digest recv: source={msg.source_uav}, seq={msg.seq}"
            )
