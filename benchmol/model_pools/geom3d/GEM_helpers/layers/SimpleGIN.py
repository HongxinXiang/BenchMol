from typing import cast

from dgl import DGLGraph, function as fn
from torch import Tensor, nn


class SimpleGIN(nn.Module):
    """
    Implementation of Graph Isomorphism Network (GIN) layer for undirected
    graphs that incorporates both node and edge features.

    This implementation does NOT transform the graph features before
    message-passing, only AFTER the message-passing are the features passed
    through a 2-layer MLP before being returned.

    This is a PyTorch + DGL equivalent of GeoGNN's `GIN`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/gnn_block.py#L75-L105
    """

    def __init__(self, in_feat_size: int, mlp_hidden_size: int, out_feat_size: int) -> None:
        """
        The "MLP" in this context is the final Multilayer Perceptron that is fed
        the graph features after message-passing, where the MLP output is then
        returned by the `forward` method.

        Args:
            in_feat_size (int): The size of each feature in the graph \
                (if the feats were encoded into embeddings, this'll be the embedding size).
            mlp_hidden_size (int): Hidden layer's size of the MLP \
                (the MLP after message-passing).
            out_feat_size (int): The output size for each feature.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, out_feat_size),
        )

    def reset_parameters(self) -> None:
        """Resets the weights of `self.mlp`."""
        cast(nn.Linear, self.mlp[0]).reset_parameters()
        cast(nn.Linear, self.mlp[2]).reset_parameters()

    def forward(self, graph: DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> Tensor:
        """
        Args:
            graph (DGLGraph): The input graph, where each node/edge feature is \
                of size `in_feat_size`, as defined in the constructor.
            node_feats (Tensor): The input node features, size `(num_of_nodes, in_feat_size)`, \
                where `in_feat_size` is defined in the constructor.
            edge_feats (Tensor): The input edge features, size `(num_of_edges, in_feat_size)`, \
                where `in_feat_size` is defined in the constructor.

        Returns:
            Tensor: Output features that incorporates both node and edge features, \
                size `(num_of_nodes, out_feat_size)`, where `out_feat_size` is \
                defined in the constructor.
        """
        with graph.local_scope():
            graph.ndata['h_n'] = node_feats
            graph.edata['h_e'] = edge_feats
            graph.update_all(
                message_func = fn.u_add_e('h_n', 'h_e', 'm'),                   # type: ignore
                reduce_func = fn.sum('m', 'h_out'),                             # type: ignore
            )
            output_node_feats = graph.ndata['h_out']
            return self.mlp.forward(output_node_feats)
