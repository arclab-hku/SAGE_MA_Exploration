import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops, to_dense_batch

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
        edge_attr: OptTensor = None,
    ):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # 本来只返回 propagate, 但是我们还需要indices
        # return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr), self.indices
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j: Tensor, edge_attr: OptTensor):
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
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        # x = self.lin1(node_feat)
        x = self.mlp1(node_feat)

        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x

class CustomTransformerConv(TransformerConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, beta=False,
                 dropout=0.0, edge_dim=None, bias=True, root_weight=True):
        super(CustomTransformerConv, self).__init__(in_channels, out_channels, heads, concat, beta, dropout, edge_dim, bias, root_weight)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: int) -> Tensor:

        if self.lin_edge is not None:
            if edge_attr is not None:
                edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
                key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

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

    def forward(self, datalist:list, viewpoint_num: List[List[int]]):
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
        # convert adj to edge_index, edge_attr and then collate them into a batch
        # batch_size = len(datalist)
        # print("batch_size", batch_size)
        # loader = DataLoader(datalist, shuffle=False, batch_size=batch_size)
        data = datalist[0]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        if self.edge_dim is None:
            edge_attr = None
        # forward pass through embedConv
        # x, indices_viewpoint = self.embed_layer(x, edge_index, edge_attr)
        x = self.embed_layer(x, edge_index, edge_attr)

        # forward pass through first transfomerConv
        x = self.active_func(self.gnn1(x, edge_index))

        # forward pass conv layers
        for i in range(len(self.gnn2)):
            x = self.active_func(self.gnn2[i](x, edge_index, edge_attr))

        # x is of shape [batch_size*num_nodes, out_channels]
        # convert to [batch_size, num_nodes, out_channels]
        # to_dense_batch 返回一个元组 (x_dense, mask)：
        # x_dense: 一个形状为 [batch_size, max_num_nodes, num_features] 的张量，表示每个图的密集节点特征表示。max_num_nodes 是所有批次中节点数最多的图的节点数。
        # mask: 一个形状为 [batch_size, max_num_nodes] 的布尔张量，用于标记有效节点的位置。有效节点对应的掩码值为 True，填充的节点对应的掩码值为 False
        
        # 如果用另一个方案, 可以考虑自己得到viewpoint_indices
        # total_node_num = x.shape[0] # 得到 batch_size*num_nodes
        # indices_viewpoint = torch.full((total_node_num), 0, dtype=torch.long)  
        
        x, mask = to_dense_batch(x, batch)
        # 另一种方案:续  
        # indices_viewpoint, indices_viewpoint_mask = to_dense_batch(indices_viewpoint, batch)
        # viewpoint_feature = x[indices_viewpoint_mask]
        
        # mask Shape: (batch_size, max_k)
        x, mask = self.gatherNodeFeats(x, viewpoint_num)  # shape [batch_size, max_viewpoint_num, out_channels]

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

    def processAdj(self, adj: Tensor):
        """
        Process adjacency matrix to filter far away nodes
        and then obtain the edge_index and edge_weight
        `adj` is of shape (batch_size, num_nodes, num_nodes)
            OR (num_nodes, num_nodes)
        """
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)
        # filter far away nodes and connection to itself
        connect_mask = ((adj < self.max_edge_dist) * (adj > 0)).float()
        adj = adj * connect_mask

        index = adj.nonzero(as_tuple=True)
        edge_attr = adj[index]

        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            index = (batch + index[1], batch + index[2])

        return torch.stack(index, dim=0), edge_attr


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
    

    def gatherNodeFeats(self, x: torch.Tensor, idx: List[List[int]], fill_value: float = 0) -> torch.Tensor:
        device = x.device  # 确保所有张量都在同一设备上
        batch_size, num_nodes, num_feats = x.shape
        max_k = max(len(sublist) for sublist in idx)  # 找到最大的子列表长度
        
        # 创建一个全为-1的索引张量，大小为(batch_size, max_k)
        idx_tensor = torch.full((batch_size, max_k), -1, dtype=torch.long, device=device)
        
        # 填充idx_tensor
        for i, sub_idx in enumerate(idx):
            if sub_idx:
                idx_tensor[i, :len(sub_idx)] = torch.tensor(sub_idx, dtype=torch.long, device=device)
        
        # 使用advanced indexing直接从x中提取特征
        # 为了处理可能的-1索引，先扩展x在节点维度上添加一个填充行
        x_padded = torch.cat([x, torch.full((batch_size, 1, num_feats), fill_value, device=device)], dim=1)
        # 使用clamp将所有-1索引改为指向新增的填充行
        gathered_nodes = x_padded[torch.arange(batch_size).unsqueeze(1), idx_tensor]

        # 创建掩码，指示有效的节点
        mask = idx_tensor != -1

        return gathered_nodes, mask.to(torch.bool)

    def graphAggr(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Aggregate the graph node features by performing global pool

        Args:
            x (Tensor): Tensor of shape [batch_size, max_num_nodes, num_feats]
            mask (Tensor): Tensor of shape [batch_size, max_num_nodes], indicating valid nodes
            aggr (str): Aggregation method for performing the global pool

        Raises:
            ValueError: If `aggr` is not in ['mean', 'max', 'add']

        Returns:
            Tensor: The global aggregated tensor of shape [batch_size, num_feats]
        """
        if self.global_aggr_type == "mean":
            # 使用mask忽略填充节点
            x = x * mask.unsqueeze(-1).float()  # 将mask应用于x
            sum_x = x.sum(dim=1)
            count = mask.sum(dim=1).unsqueeze(-1).float()
            return sum_x / count

        elif self.global_aggr_type == "max":
            # 使用一个非常小的值来忽略填充节点
            x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            max_feats, _ = x.max(dim=1)
            return max_feats

        elif self.global_aggr_type == "add":
            # 使用mask忽略填充节点
            x = x * mask.unsqueeze(-1).float()
            return x.sum(dim=1)

        else:
            raise ValueError(f"`aggr` should be one of 'mean', 'max', 'add'")


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

    def forward(self, node_obs: Tensor, viewpoint_num: Tensor):
        x, mask = self.gnn(node_obs, viewpoint_num)
        return x, mask

    @property
    def out_dim(self):
        return self.hidden_size + (self.heads - 1) * self.concat * (self.hidden_size)
