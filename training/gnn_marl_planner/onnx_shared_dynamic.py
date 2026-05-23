import os
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch


class DynamicPolicyHeadModule(torch.nn.Module):
    """ONNX-exportable policy head with dynamic candidate count."""

    def __init__(self, score_linear: torch.nn.Linear, mix_linear: torch.nn.Linear):
        super().__init__()
        self.score_linear = torch.nn.Linear(
            score_linear.in_features, score_linear.out_features, bias=True
        )
        self.mix_linear = torch.nn.Linear(
            mix_linear.in_features, mix_linear.out_features, bias=True
        )
        with torch.no_grad():
            self.score_linear.weight.copy_(score_linear.weight)
            self.score_linear.bias.copy_(score_linear.bias)
            self.mix_linear.weight.copy_(mix_linear.weight)
            self.mix_linear.bias.copy_(mix_linear.bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        aggr_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # candidate_features: [B, K, F], masks: [B, K]
        logits = self.score_linear(candidate_features).squeeze(-1)

        candidate_valid = candidate_mask > 0
        neighbor_valid = neighbor_mask > 0
        neg_inf = torch.full_like(logits, -1.0e4)

        primary_logits = torch.where(candidate_valid, logits, neg_inf)
        neighbor_logits = torch.where(neighbor_valid, logits, neg_inf)

        primary_probs = torch.softmax(primary_logits, dim=-1)
        neighbor_probs = torch.softmax(neighbor_logits, dim=-1)

        mix_ratio = torch.sigmoid(self.mix_linear(aggr_features))  # [B, 1]
        mixed_probs = (1.0 - mix_ratio) * primary_probs + mix_ratio * neighbor_probs

        # Keep invalid actions at zero and re-normalize each row.
        mixed_probs = mixed_probs * candidate_valid.to(mixed_probs.dtype)
        denom = mixed_probs.sum(dim=-1, keepdim=True)

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
            denom > 0,
            mixed_probs / torch.clamp(denom, min=1.0e-12),
            fallback,
        )
        return mixed_probs, mix_ratio


class SharedDynamicOnnxPolicy:
    """Runs dynamic-K ONNX policy head on top of shared actor base features."""

    def __init__(
        self,
        shared_actor: torch.nn.Module,
        onnx_model_path: str,
        force_export: bool = False,
    ):
        if shared_actor is None:
            raise ValueError("shared_actor must be initialized before ONNX policy init")

        action_out = shared_actor.act.action_out
        score_linear = action_out.linear
        mix_linear = action_out.policy_mix_ratio_nn

        self._policy_head = DynamicPolicyHeadModule(score_linear, mix_linear).eval()
        self.onnx_model_path = onnx_model_path

        os.makedirs(os.path.dirname(self.onnx_model_path), exist_ok=True)
        if force_export or (not os.path.isfile(self.onnx_model_path)):
            self.export_dynamic_onnx(self.onnx_model_path)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(self.onnx_model_path, session_options)

    @property
    def feature_dim(self) -> int:
        return int(self._policy_head.score_linear.in_features)

    def export_dynamic_onnx(self, onnx_model_path: str) -> None:
        feat_dim = self.feature_dim
        dummy_inputs = (
            torch.zeros((2, 4, feat_dim), dtype=torch.float32),
            torch.zeros((2, feat_dim), dtype=torch.float32),
            torch.ones((2, 4), dtype=torch.int64),
            torch.ones((2, 4), dtype=torch.int64),
        )
        torch.onnx.export(
            self._policy_head,
            dummy_inputs,
            onnx_model_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[
                "candidate_features",
                "aggr_features",
                "candidate_mask",
                "neighbor_mask",
            ],
            output_names=["mixed_probs", "mix_ratio"],
            dynamic_axes={
                "candidate_features": {0: "batch_size", 1: "num_candidates"},
                "aggr_features": {0: "batch_size"},
                "candidate_mask": {0: "batch_size", 1: "num_candidates"},
                "neighbor_mask": {0: "batch_size", 1: "num_candidates"},
                "mixed_probs": {0: "batch_size", 1: "num_candidates"},
                "mix_ratio": {0: "batch_size"},
            },
        )

    def run_policy_inputs(
        self,
        candidate_features: torch.Tensor,
        aggr_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs = {
            "candidate_features": candidate_features.detach()
            .cpu()
            .to(torch.float32)
            .numpy(),
            "aggr_features": aggr_features.detach().cpu().to(torch.float32).numpy(),
            "candidate_mask": candidate_mask.detach().cpu().to(torch.int64).numpy(),
            "neighbor_mask": neighbor_mask.detach().cpu().to(torch.int64).numpy(),
        }
        mixed_probs, mix_ratio = self.session.run(None, inputs)
        return mixed_probs, mix_ratio

    @staticmethod
    def select_actions(
        mixed_probs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
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
        shared_actor: torch.nn.Module,
        graph_list: Sequence[object],
        indices_viewpoint_batch: List[List[int]],
        all_indices_viewpoint_batch: List[List[int]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            actor_features, aggr_actor_feature, masks, neighbor_masks, _, _ = shared_actor.base(
                list(graph_list),
                indices_viewpoint_batch,
                all_indices_viewpoint_batch,
            )
        mixed_probs, mix_ratio = self.run_policy_inputs(
            actor_features, aggr_actor_feature, masks, neighbor_masks
        )
        actions = self.select_actions(mixed_probs, deterministic=deterministic)
        return actions, mixed_probs, mix_ratio
