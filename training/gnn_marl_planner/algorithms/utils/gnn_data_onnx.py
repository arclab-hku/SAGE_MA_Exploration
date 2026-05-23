import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops

import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

from .util import init, get_clones
import math
from torch_geometric.utils import softmax

"""GNN modules"""


class EmbedConv(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
        use_layerNorm: bool,
        add_self_loop: bool,
        edge_dim: int = 0,
    ):
        """
            EmbedConv Layer which takes in node features, node_type (entity type)
            and the  edge features (if they exist)
            `entity_embedding` is concatenated with `node_features` and
            `edge_features` and are passed through linear layers.
            The `message_passing` is similar to GCN layer

        Args:
            input_dim (int):
                The node feature dimension
            num_embeddings (int): 和mlp的input_dim不同,他不影响网络的结构,只是告诉我们需要embedding的类型的数量,
                在这个环境中,就是node_type的数量
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the linear layers
            layer_N (int):
                Number of linear layers for aggregation
            use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer
            use_ReLU (bool):
                Whether to use reLU for each layer
            use_layerNorm (bool):
                Whether to use layerNorm for each layer
            add_self_loop (bool):
                Whether to add self loops in the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered. Defaults to 0.
        """
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        layer_norm = [nn.Identity(), nn.LayerNorm(hidden_size)][use_layerNorm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])
        self.indices = None
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        print("num_embeddings", num_embeddings)
        print("embedding_size", embedding_size)
        self.entity_embed = nn.Embedding(1000, embedding_size)
        print("input_dim + embedding_size + edge_dim is", input_dim + embedding_size + edge_dim)
        self.mlp1 = nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)
        
        self.lin1 = nn.Sequential(
            init_(nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)),
            active_func,
            layer_norm,
        )
        self.lin_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), active_func, layer_norm
        )

        self.lin2 = get_clones(self.lin_h, self._layer_N)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
    ):
        # if self._add_self_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # 本来只返回 propagate, 但是我们还需要indices
        # return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr), self.indices
        return self.propagate(edge_index=edge_index, x=x)
    
    def message(self, x_j: Tensor):
        """
        The node_obs obtained from the environment
        is actually [node_features, node_num, entity_type]
        x_i' = AGG([x_j, EMB(ent_j), e_ij] : j \in \mathcal{N}(i))
        """
        node_feat_j = x_j[:, :-1]
        # dont forget to convert to torch.LongTensor
        entity_type_j = x_j[:, -1].long()
        # self.indices = torch.nonzero(entity_type_j == 1).squeeze()

        entity_embed_j = self.entity_embed(entity_type_j)

        node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        # x = self.lin1(node_feat)
        x = self.mlp1(node_feat)

        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x

class CustomTransformerConv(TransformerConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, beta=False,
                 dropout=0.0, edge_dim=None, bias=True, root_weight=True):
        super(CustomTransformerConv, self).__init__(in_channels=-1, out_channels=out_channels, heads=heads, concat=concat, beta=beta, dropout=dropout, edge_dim=None, bias=bias, root_weight=root_weight)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: int) -> Tensor:


        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j


        out = out * alpha.view(-1, self.heads, 1)
        return out

class TransformerConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        num_heads: int,
        concat_heads: bool,
        layer_N: int,
        use_ReLU: bool,
        # graph_aggr: str,
        global_aggr_type: str,
        embed_hidden_size: int,
        embed_layer_N: int,
        embed_use_orthogonal: bool,
        embed_use_ReLU: bool,
        embed_use_layerNorm: bool,
        embed_add_self_loop: bool,
        # max_edge_dist: float,
        edge_dim: int = 1,
    ):
        """
            Module for Transformer Graph Conv Net:
            • This will process the adjacency weight matrix, construct the binary
                adjacency matrix according to `max_edge_dist` parameter, assign
                edge weights as the weights in the adjacency weight matrix.
            • After this, the batch data is converted to a PyTorch Geometric
                compatible dataloader.
            • Then the batch is passed through the graph neural network.
            • The node feature output is then either:
                • Aggregated across the graph to get graph encoded data.
                • Pull node specific `message_passed` hidden feature as output.

        Args:
            input_dim (int):
                Node feature dimension
                NOTE: a reduction of `input_dim` by 1 will be carried out
                internally because `node_obs` = [node_feat, entity_type]
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the attention layers
            num_heads (int):
                Number of heads in the attention layer
            concat_heads (bool):
                Whether to concatenate the heads in the attention layer or
                average them
            layer_N (int):
                Number of attention layers for aggregation
            use_ReLU (bool):
                Whether to use reLU for each layer
            graph_aggr (str):
                Whether we want to pull node specific features from the output or
                perform global_pool on all nodes.
                Choices: ['global', 'node']
            global_aggr_type (str):
                The type of aggregation to perform if `graph_aggr` is `global`
                Choices: ['mean', 'max', 'add']
            embed_hidden_size (int):
                Hidden layer size of the linear layers in `EmbedConv`
            embed_layer_N (int):
                Number of linear layers for aggregation in `EmbedConv`
            embed_use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer in `EmbedConv`
            embed_use_ReLU (bool):
                Whether to use reLU for each layer in `EmbedConv`
            embed_use_layerNorm (bool):
                Whether to use layerNorm for each layer in `EmbedConv`
            embed_add_self_loop (bool):
                Whether to add self loops in the graph in `EmbedConv`
            max_edge_dist (float):
                The maximum edge distance to consider while constructing the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered in `EmbedConv`. Defaults to 1.
        """
        super(TransformerConvNet, self).__init__()
        self.active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        # self.max_edge_dist = max_edge_dist
        # self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type
        # NOTE: reducing dimension of input by 1 because
        # node_obs = [node_feat, entity_type]
        self.embed_layer = EmbedConv(
            input_dim=input_dim - 1,
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
            add_self_loop=embed_add_self_loop,
            edge_dim=edge_dim,
        )
        self.gnn1 = CustomTransformerConv(
            in_channels=embed_hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
            concat=concat_heads,
            beta=False,
            dropout=0.0,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True,
        )
        self.gnn2 = nn.ModuleList()
        for i in range(layer_N):
            self.gnn2.append(
                self.addTCLayer(self.getInChannels(hidden_size), hidden_size)
            )

    def forward(self, x, edge_index, viewpoint_idx, viewpoint_mask=None):
        """
        node_obs: Tensor shape:(batch_size, num_nodes, node_obs_dim)
            Node features in the graph formed wrt agent_i
        adj: Tensor shape:(batch_size, num_nodes, num_nodes)
            Adjacency Matrix for the graph formed wrt agent_i
            NOTE: Right now the adjacency matrix is the distance
            magnitude between all entities so will have to post-process
            this to obtain the edge_index and edge_attr
        agent_id: Tensor shape:(batch_size) or (batch_size, k)
            Node number for agent_i in the graph. This will be used to
            pull out the aggregated features from that node
        """
        # forward pass through embedConv
        # x, indices_viewpoint = self.embed_layer(x, edge_index, edge_attr)
        x = self.embed_layer(x, edge_index)

        # forward pass through first transfomerConv
        x = self.active_func(self.gnn1(x, edge_index))
        # forward pass conv layers
        for i in range(len(self.gnn2)):
            x = self.active_func(self.gnn2[i](x, edge_index))

        # ONNX path uses explicit candidate indices/masks (tensor-only).
        x, mask = self.gatherNodeFeatsTensor(x, viewpoint_idx, viewpoint_mask)
        return x, mask
    def reshape_viewpoint_features(self, x, indices_viewpoint_mask, fill_value=0.0):
        batch_size, num_nodes, feature_dim = x.shape
        viewpoint_num = indices_viewpoint_mask.sum(dim=1).max().item()  # 最大的 True 数量
        padded_viewpoint_features = torch.full((batch_size, viewpoint_num, feature_dim), fill_value=fill_value, dtype=x.dtype, device=x.device)

        for i in range(batch_size):
            valid_indices = indices_viewpoint_mask[i].nonzero(as_tuple=False).squeeze()
            valid_features = x[i][valid_indices]
            padded_viewpoint_features[i, :valid_features.size(0)] = valid_features

        return padded_viewpoint_features
    def addTCLayer(self, in_channels: int, out_channels: int):
        """
        Add TransformerConv Layer

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Returns:
            TransformerConv: returns a TransformerConv Layer
        """
        return CustomTransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=self.num_heads,
            concat=self.concat_heads,
            beta=False,
            dropout=0.0,
            edge_dim=self.edge_dim,
            root_weight=True,
        )

    def getInChannels(self, out_channels: int):
        """
        Given the out_channels of the previous layer return in_channels
        for the next layer. This depends on the number of heads and whether
        we are concatenating the head outputs
        """
        return out_channels + (self.num_heads - 1) * self.concat_heads * (out_channels)

    # def gatherNodeFeats(self, x: torch.Tensor, idx: List[List[int]], fill_value: float = 0) -> torch.Tensor:
    #     """
    #     Args:
    #         x (Tensor): Tensor of shape (batch_size, num_nodes, out_channels)
    #         idx (List[List[int]]): List of lists, where each sublist contains the indices of nodes to pull from the graph.
    #         fill_value (float): The value to fill for padding in the output tensor.

    #     Returns:
    #         Tensor: Tensor of shape (batch_size, max_k, out_channels) which contains the features from the nodes of interest,
    #                 padded with fill_value where necessary.
    #     """
    #     out = []
    #     mask = []
    #     batch_size, num_nodes, num_feats = x.shape
    #     max_k = max(len(sublist) for sublist in idx)  # Find the maximum length of sublists in idx
    #     idx_tensor = [torch.tensor(sub_idx).to(self.device) for sub_idx in idx]  # 将每个子列表转换为张量并移动到正确的设备

    #     for batch_idx in range(batch_size):
    #         gathered_nodes = []
    #         mask_row = []
    #         for node_idx in idx_tensor[batch_idx]:
    #             node_feat = x[batch_idx, node_idx].unsqueeze(0)  # Shape: (1, num_feats)
    #             gathered_nodes.append(node_feat)
    #             mask_row.append(1)  # Mark this position as valid
    #         if len(gathered_nodes) < max_k:  # Padding if necessary
    #             pad_len = max_k - len(gathered_nodes)
    #             padding = torch.full((pad_len, num_feats), fill_value, dtype=x.dtype)
    #             gathered_nodes.append(padding)
    #             mask_row.extend([0] * pad_len)   # Mark this position as padding 
    #         gathered_nodes = torch.cat(gathered_nodes, dim=0)  # Shape: (max_k, num_feats)
    #         out.append(gathered_nodes.unsqueeze(0))  # Shape: (1, max_k, num_feats)
    #         mask.append(mask_row)

    #     out = torch.cat(out, dim=0)  # Shape: (batch_size, max_k, num_feats)
    #     mask = torch.tensor(mask, dtype=torch.bool)  # Shape: (batch_size, max_k)
    #     return out, mask
    

    def gatherNodeFeatsTensor(
        self,
        x: torch.Tensor,
        idx_tensor: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        fill_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather candidate node features from a single graph with tensor indices.

        Args:
            x: Node features, shape [N, F].
            idx_tensor: Candidate node indices, shape [B, K] or [K].
            idx_mask: Valid mask for candidate indices, shape [B, K] or [K].
        Returns:
            gathered: [B, K, F]
            mask: [B, K] bool
        """
        if idx_tensor.dim() == 1:
            idx_tensor = idx_tensor.unsqueeze(0)
        idx_tensor = idx_tensor.to(dtype=torch.long, device=x.device)

        if idx_mask is None:
            idx_mask = (idx_tensor >= 0).to(torch.int64)
        elif idx_mask.dim() == 1:
            idx_mask = idx_mask.unsqueeze(0)
        idx_mask = idx_mask.to(dtype=torch.int64, device=x.device)

        B, K = idx_tensor.shape
        Fdim = x.shape[1]
        pad_row = torch.full((1, Fdim), fill_value, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, pad_row], dim=0)  # [N+1, F]
        pad_idx = x.shape[0]

        safe_idx = torch.where(idx_mask > 0, idx_tensor, torch.full_like(idx_tensor, pad_idx))
        safe_idx = torch.clamp(safe_idx, min=0, max=pad_idx)
        gathered = x_padded.index_select(0, safe_idx.reshape(-1)).reshape(B, K, Fdim)
        mask = idx_mask > 0
        return gathered, mask

class GNNBase(nn.Module):
    """
    A Wrapper for constructing the Base graph neural network.
    This uses TransformerConv from Pytorch Geometric
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
    and embedding layers for entity types
    Params:
    args: (argparse.Namespace)
        Should contain the following arguments
        num_embeddings: (int)
            Number of entity types in the env to have different embeddings
            for each entity type
        embedding_size: (int)
            Embedding layer output size for each entity category
        embed_hidden_size: (int)
            Hidden layer dimension after the embedding layer
        embed_layer_N: (int)
            Number of hidden linear layers after the embedding layer")
        embed_use_ReLU: (bool)
            Whether to use ReLU in the linear layers after the embedding layer
        embed_add_self_loop: (bool)
            Whether to add self loops in adjacency matrix
        gnn_hidden_size: (int)
            Hidden layer dimension in the GNN
        gnn_num_heads: (int)
            Number of heads in the transformer conv layer (GNN)
        gnn_concat_heads: (bool)
            Whether to concatenate the head output or average
        gnn_layer_N: (int)
            Number of GNN conv layers
        gnn_use_ReLU: (bool)
            Whether to use ReLU in GNN conv layers
        max_edge_dist: (float)
            Maximum distance above which edges cannot be connected between
            the entities
        graph_feat_type: (str)
            Whether to use 'global' node/edge feats or 'relative'
            choices=['global', 'relative']
    node_obs_shape: (Union[Tuple, List])
        The node observation shape. Example: (18,)
    edge_dim: (int)
        Dimensionality of edge attributes
    """

    def __init__(
        self,
        args: argparse.Namespace,
        node_obs_shape: int,
        edge_dim: int,
        # graph_aggr: str,
    ):
        super(GNNBase, self).__init__()

        self.args = args
        self.hidden_size = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads
        self.gnn = TransformerConvNet(
            input_dim=node_obs_shape,
            edge_dim=edge_dim,
            num_embeddings=args.num_embeddings,
            embedding_size=args.embedding_size,
            hidden_size=args.gnn_hidden_size,
            num_heads=args.gnn_num_heads,
            concat_heads=args.gnn_concat_heads,
            layer_N=args.gnn_layer_N,
            use_ReLU=args.gnn_use_ReLU,
            # graph_aggr=graph_aggr,
            global_aggr_type=args.global_aggr_type,
            embed_hidden_size=args.embed_hidden_size,
            embed_layer_N=args.embed_layer_N,
            embed_use_orthogonal=args.use_orthogonal,
            embed_use_ReLU=args.embed_use_ReLU,
            embed_use_layerNorm=args.use_feature_normalization,
            embed_add_self_loop=args.embed_add_self_loop,
            # max_edge_dist=args.max_edge_dist,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        viewpoint_idx: Tensor,
        viewpoint_mask: Optional[Tensor] = None,
    ):
        dummy_use = viewpoint_idx.float().sum() * 0.0
        x = x + dummy_use  # 添加一个不影响结果的计算操作
        x, mask = self.gnn(x, edge_index, viewpoint_idx, viewpoint_mask)
        return x, mask

    @property
    def out_dim(self):
        return self.hidden_size + (self.heads - 1) * self.concat * (self.hidden_size)
