import copy
import math
import os
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


def _target_matrix(dst: torch.Tensor, num_nodes) -> torch.Tensor:
    """Return one-hot target assignment matrix T, shape [E, N]."""
    nodes = torch.arange(num_nodes, device=dst.device, dtype=torch.long)
    return (dst.unsqueeze(1) == nodes.unsqueeze(0)).to(torch.float32)


class DenseEmbedConvFromPyg(nn.Module):
    """ONNX-friendly dense version of EmbedConv message passing."""

    def __init__(self, embed_layer: nn.Module):
        super().__init__()
        self.add_self_loops = bool(getattr(embed_layer, "_add_self_loops", False))
        self.entity_embed = copy.deepcopy(embed_layer.entity_embed)
        self.mlp1 = copy.deepcopy(embed_layer.mlp1)
        self.lin2 = nn.ModuleList([copy.deepcopy(m) for m in embed_layer.lin2])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        if self.add_self_loops:
            diag = torch.arange(num_nodes, device=x.device, dtype=torch.long)
            src = torch.cat([src, diag], dim=0)
            dst = torch.cat([dst, diag], dim=0)
        target = _target_matrix(dst, num_nodes)  # [E, N]

        node_feat = x[:, :-1]
        entity_type = x[:, -1].to(torch.long)
        emb = self.entity_embed(entity_type)
        msg = torch.cat([node_feat, emb], dim=1)
        msg = self.mlp1(msg)
        for layer in self.lin2:
            msg = layer(msg)
        msg_src = msg.index_select(0, src)  # [E, F]
        return target.transpose(0, 1) @ msg_src  # [N, F]


class DenseTransformerConvFromPyg(nn.Module):
    """ONNX-friendly dense equivalent of CustomTransformerConv."""

    def __init__(self, conv: nn.Module):
        super().__init__()
        self.heads = int(conv.heads)
        self.out_channels = int(conv.out_channels)
        self.concat = bool(conv.concat)
        self.root_weight = bool(conv.root_weight)

        self.lin_query = copy.deepcopy(conv.lin_query)
        self.lin_key = copy.deepcopy(conv.lin_key)
        self.lin_value = copy.deepcopy(conv.lin_value)
        self.lin_skip = copy.deepcopy(conv.lin_skip) if self.root_weight else None
        self.lin_beta = copy.deepcopy(conv.lin_beta) if conv.lin_beta is not None else None
        if hasattr(conv, "log_temperature"):
            self.log_temperature = nn.Parameter(conv.log_temperature.detach().clone())
        else:
            self.log_temperature = nn.Parameter(torch.zeros(self.heads, dtype=torch.float32))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        target = _target_matrix(dst, num_nodes)  # [E, N]
        target_t = target.transpose(0, 1)  # [N, E]

        h = self.heads
        c = self.out_channels
        query = self.lin_query(x).reshape(-1, h, c).index_select(0, dst)  # [E, H, C]
        key = self.lin_key(x).reshape(-1, h, c).index_select(0, src)  # [E, H, C]
        value = self.lin_value(x).reshape(-1, h, c).index_select(0, src)  # [E, H, C]

        raw_dots = (query * key).sum(dim=-1) / math.sqrt(float(c))  # [E, H]
        temperature = torch.exp(self.log_temperature).view(1, -1)  # [1, H]
        scores = raw_dots * temperature
        scores_he = scores.transpose(0, 1)  # [H, E]

        # Group softmax over incoming edges per dst node, per head.
        scores_hne = scores_he.unsqueeze(1).expand(-1, num_nodes, -1)  # [H, N, E]
        mask_hne = target_t.unsqueeze(0) > 0
        neg_inf = torch.full_like(scores_hne, -1.0e4)
        masked_scores = torch.where(mask_hne, scores_hne, neg_inf)
        max_hn = masked_scores.max(dim=-1).values  # [H, N]
        max_he = max_hn @ target_t  # [H, E]

        exp_he = torch.exp(scores_he - max_he)
        sum_hn = exp_he @ target  # [H, N]
        sum_he = sum_hn @ target_t  # [H, E]
        alpha_he = exp_he / torch.clamp(sum_he, min=1.0e-12)  # [H, E]
        alpha_eh = alpha_he.transpose(0, 1)  # [E, H]

        # Keep the same skip-softmax gradient path used in the deployed model.
        direct_lambda = 0.5
        direct_alpha = 0.5 * (1.0 + torch.tanh(raw_dots * 0.3))
        diversity_bonus = 0.1 * torch.clamp(torch.abs(raw_dots), max=1.0)
        effective_alpha = alpha_eh + direct_lambda * direct_alpha + diversity_bonus

        weighted = effective_alpha.unsqueeze(-1) * value  # [E, H, C]
        out = torch.einsum("ehc,en->nhc", weighted, target)  # [N, H, C]
        if self.concat:
            out = out.reshape(-1, h * c)
        else:
            out = out.mean(dim=1)

        if self.root_weight and self.lin_skip is not None:
            x_r = self.lin_skip(x)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1)).sigmoid()
                out = beta * x_r + (1.0 - beta) * out
            else:
                out = out + x_r
        return out


class DenseGNNBaseFromSharedActor(nn.Module):
    """ONNX-friendly graph encoder cloned from shared_actor.base weights."""

    def __init__(self, shared_actor: nn.Module):
        super().__init__()
        src_base = shared_actor.base
        if not hasattr(src_base, "gnn"):
            raise RuntimeError("shared_actor.base.gnn is required")
        src_gnn = src_base.gnn
        self.active_func = copy.deepcopy(src_gnn.active_func)
        self.embed = DenseEmbedConvFromPyg(src_gnn.embed_layer)
        self.gnn1 = DenseTransformerConvFromPyg(src_gnn.gnn1)
        self.gnn2 = nn.ModuleList([DenseTransformerConvFromPyg(layer) for layer in src_gnn.gnn2])

    @staticmethod
    def gather_node_feats(
        x: torch.Tensor,
        idx_tensor: torch.Tensor,
        idx_mask: torch.Tensor,
        fill_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx_tensor.dim() == 1:
            idx_tensor = idx_tensor.unsqueeze(0)
        if idx_mask.dim() == 1:
            idx_mask = idx_mask.unsqueeze(0)
        idx_tensor = idx_tensor.to(dtype=torch.long, device=x.device)
        idx_mask = idx_mask.to(dtype=torch.int64, device=x.device)

        batch, k = idx_tensor.shape
        feat_dim = x.shape[1]
        pad = torch.full((1, feat_dim), fill_value, dtype=x.dtype, device=x.device)
        x_pad = torch.cat([x, pad], dim=0)
        pad_idx = x.shape[0]

        safe_idx = torch.where(idx_mask > 0, idx_tensor, torch.full_like(idx_tensor, pad_idx))
        safe_idx = torch.clamp(safe_idx, min=0, max=pad_idx)
        gathered = x_pad.index_select(0, safe_idx.reshape(-1)).reshape(batch, k, feat_dim)
        mask = idx_mask > 0
        return gathered, mask

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        viewpoint_idx: torch.Tensor,
        viewpoint_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x, edge_index)
        x = self.active_func(self.gnn1(x, edge_index))
        for layer in self.gnn2:
            x = self.active_func(layer(x, edge_index))
        return self.gather_node_feats(x, viewpoint_idx, viewpoint_mask)


class FullDynamicOnnxModel(nn.Module):
    """Full graph encoder + policy head model for ONNX export."""

    def __init__(
        self,
        base: DenseGNNBaseFromSharedActor,
        score_linear: nn.Linear,
        mix_linear: nn.Linear,
    ):
        super().__init__()
        self.base = base
        self.score_linear = nn.Linear(
            score_linear.in_features, score_linear.out_features, bias=True
        )
        self.mix_linear = nn.Linear(
            mix_linear.in_features, mix_linear.out_features, bias=True
        )
        with torch.no_grad():
            self.score_linear.weight.copy_(score_linear.weight)
            self.score_linear.bias.copy_(score_linear.bias)
            self.mix_linear.weight.copy_(mix_linear.weight)
            self.mix_linear.bias.copy_(mix_linear.bias)

    @property
    def feature_dim(self) -> int:
        return int(self.score_linear.in_features)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        candidate_idx: torch.Tensor,
        candidate_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        candidate_features, _ = self.base(
            x=x,
            edge_index=edge_index,
            viewpoint_idx=candidate_idx,
            viewpoint_mask=candidate_mask,
        )  # [B, K, F]

        candidate_valid = candidate_mask > 0
        neighbor_valid = neighbor_mask > 0
        valid_f = candidate_valid.to(candidate_features.dtype).unsqueeze(-1)

        denom = torch.clamp(valid_f.sum(dim=1), min=1.0)
        aggr_features = (candidate_features * valid_f).sum(dim=1) / denom

        logits = self.score_linear(candidate_features).squeeze(-1)  # [B, K]
        neg_inf = torch.full_like(logits, -1.0e4)

        primary_logits = torch.where(candidate_valid, logits, neg_inf)
        neighbor_logits = torch.where(neighbor_valid, logits, neg_inf)

        primary_probs = torch.softmax(primary_logits, dim=-1)
        neighbor_probs = torch.softmax(neighbor_logits, dim=-1)

        mix_ratio = torch.sigmoid(self.mix_linear(aggr_features))  # [B, 1]
        mixed_probs = (1.0 - mix_ratio) * primary_probs + mix_ratio * neighbor_probs

        mixed_probs = mixed_probs * candidate_valid.to(mixed_probs.dtype)
        denom_probs = mixed_probs.sum(dim=-1, keepdim=True)

        fallback = candidate_valid.to(mixed_probs.dtype)
        fallback_denom = fallback.sum(dim=-1, keepdim=True)
        uniform = torch.ones_like(fallback)
        uniform = uniform / torch.clamp(uniform.sum(dim=-1, keepdim=True), min=1.0)
        fallback = torch.where(
            fallback_denom > 0,
            fallback / torch.clamp(fallback_denom, min=1.0),
            uniform,
        )

        mixed_probs = torch.where(
            denom_probs > 0,
            mixed_probs / torch.clamp(denom_probs, min=1.0e-12),
            fallback,
        )
        return mixed_probs, mix_ratio


class FullDynamicOnnxPolicy:
    """Run full graph model in ONNXRuntime with dynamic N/E/K."""

    def __init__(
        self,
        shared_actor: torch.nn.Module,
        onnx_model_path: str,
        force_export: bool = False,
    ):
        if shared_actor is None:
            raise ValueError("shared_actor must be initialized before full ONNX init")

        action_out = shared_actor.act.action_out
        if not hasattr(action_out, "linear") or not hasattr(action_out, "policy_mix_ratio_nn"):
            raise RuntimeError("shared_actor.act.action_out must expose linear and policy_mix_ratio_nn")

        base = DenseGNNBaseFromSharedActor(shared_actor).eval()
        self.model = FullDynamicOnnxModel(
            base=base,
            score_linear=action_out.linear,
            mix_linear=action_out.policy_mix_ratio_nn,
        ).eval()

        self.onnx_model_path = onnx_model_path
        os.makedirs(os.path.dirname(self.onnx_model_path), exist_ok=True)
        if force_export or (not os.path.isfile(self.onnx_model_path)):
            self.export_dynamic_onnx(self.onnx_model_path)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(self.onnx_model_path, session_options)

    @property
    def feature_dim(self) -> int:
        return self.model.feature_dim

    def export_dynamic_onnx(self, onnx_model_path: str) -> None:
        dummy_inputs = (
            torch.zeros((32, 6), dtype=torch.float32),  # x [N, 6]
            torch.zeros((2, 96), dtype=torch.int64),  # edge_index [2, E]
            torch.zeros((1, 8), dtype=torch.int64),  # candidate_idx [B, K]
            torch.ones((1, 8), dtype=torch.int64),  # candidate_mask [B, K]
            torch.ones((1, 8), dtype=torch.int64),  # neighbor_mask [B, K]
        )
        dummy_inputs[0][:, -1] = torch.randint(0, 5, (32,), dtype=torch.int64).to(torch.float32)

        torch.onnx.export(
            self.model,
            dummy_inputs,
            onnx_model_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[
                "x",
                "edge_index",
                "candidate_idx",
                "candidate_mask",
                "neighbor_mask",
            ],
            output_names=["mixed_probs", "mix_ratio"],
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "candidate_idx": {0: "batch_size", 1: "num_candidates"},
                "candidate_mask": {0: "batch_size", 1: "num_candidates"},
                "neighbor_mask": {0: "batch_size", 1: "num_candidates"},
                "mixed_probs": {0: "batch_size", 1: "num_candidates"},
                "mix_ratio": {0: "batch_size"},
            },
        )

    def run_policy_inputs(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        candidate_idx: np.ndarray,
        candidate_mask: np.ndarray,
        neighbor_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs = {
            "x": x.astype(np.float32, copy=False),
            "edge_index": edge_index.astype(np.int64, copy=False),
            "candidate_idx": candidate_idx.astype(np.int64, copy=False),
            "candidate_mask": candidate_mask.astype(np.int64, copy=False),
            "neighbor_mask": neighbor_mask.astype(np.int64, copy=False),
        }
        mixed_probs, mix_ratio = self.session.run(None, inputs)
        return mixed_probs, mix_ratio

    @staticmethod
    def select_actions(mixed_probs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return np.argmax(mixed_probs, axis=1).astype(np.int64).reshape(-1, 1)

        out = np.zeros((mixed_probs.shape[0], 1), dtype=np.int64)
        for i in range(mixed_probs.shape[0]):
            row = mixed_probs[i]
            denom = float(row.sum())
            if denom <= 0.0 or not np.isfinite(denom):
                out[i, 0] = 0
                continue
            p = row / denom
            out[i, 0] = int(np.random.choice(len(p), p=p))
        return out

    def infer_from_graphs(
        self,
        graph_list: Sequence[object],
        indices_viewpoint_batch: List[List[int]],
        all_indices_viewpoint_batch: List[List[int]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions_rows: List[np.ndarray] = []
        probs_rows: List[np.ndarray] = []
        mix_rows: List[np.ndarray] = []
        max_k = 1

        for i, graph in enumerate(graph_list):
            if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
                raise ValueError(f"graph_list[{i}] is not a PyG Data-like object")
            x = graph.x.detach().cpu().to(torch.float32).numpy()
            edge_index = graph.edge_index.detach().cpu().to(torch.int64).numpy()

            cand = [int(v) for v in (indices_viewpoint_batch[i] if i < len(indices_viewpoint_batch) else [])]
            neigh = [int(v) for v in (all_indices_viewpoint_batch[i] if i < len(all_indices_viewpoint_batch) else [])]

            if not cand:
                candidate_idx = np.zeros((1, 1), dtype=np.int64)
                candidate_mask = np.zeros((1, 1), dtype=np.int64)
                neighbor_mask = np.zeros((1, 1), dtype=np.int64)
            else:
                k = len(cand)
                candidate_idx = np.asarray(cand, dtype=np.int64).reshape(1, k)
                candidate_mask = np.ones((1, k), dtype=np.int64)
                # Align with the currently trained PyTorch path:
                # gatherNodeFeats() collapses neighbor_mask to mask.clone(),
                # so the exported full-ONNX policy should not introduce an
                # extra neighbor-only restriction here.
                neighbor_mask = candidate_mask.copy()

            mixed_probs, mix_ratio = self.run_policy_inputs(
                x=x,
                edge_index=edge_index,
                candidate_idx=candidate_idx,
                candidate_mask=candidate_mask,
                neighbor_mask=neighbor_mask,
            )
            action = self.select_actions(mixed_probs, deterministic=deterministic)

            actions_rows.append(action.reshape(1, 1))
            probs_rows.append(mixed_probs.reshape(1, -1))
            mix_rows.append(mix_ratio.reshape(1, -1))
            max_k = max(max_k, int(mixed_probs.shape[1]))

        batch = len(actions_rows)
        actions = np.zeros((batch, 1), dtype=np.int64)
        mixed_probs_padded = np.zeros((batch, max_k), dtype=np.float32)
        mix_ratio = np.zeros((batch, 1), dtype=np.float32)
        for i in range(batch):
            actions[i, 0] = int(actions_rows[i][0, 0])
            row = probs_rows[i][0]
            mixed_probs_padded[i, : row.shape[0]] = row
            mix_ratio[i, 0] = float(mix_rows[i][0, 0])
        return actions, mixed_probs_padded, mix_ratio
