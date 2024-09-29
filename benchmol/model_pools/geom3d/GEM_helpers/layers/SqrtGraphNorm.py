import torch
from dgl import DGLGraph
from torch import Tensor


class SqrtGraphNorm(torch.nn.Module):
    """
    Applies graph normalization, where each node features is divided by
    sqrt(num_of_nodes) for each graph in batched graph created by `dgl.batch`.

    This is a PyTorch + DGL equivalent of GeoGNN's `GraphNorm`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/gnn_block.py#L26
    """

    def forward(self, batched_graph: DGLGraph, node_feats: Tensor) -> Tensor:
        """
        Args:
            batched_graph (DGLGraph): Batched (or unbatched) DGL graph created by `dgl.batch` (or `dgl.graph` for unbatched).
            node_feats (Tensor): The input node features.

        Returns:
            Tensor: The node features that's been normalized via dividing by sqrt(num_of_nodes) for each graph.
        """
        batch_num_of_nodes = batched_graph.batch_num_nodes()
        assert isinstance(batch_num_of_nodes, Tensor)

        norm_factors = torch.sqrt(batch_num_of_nodes.float())
        norm_factors = norm_factors.repeat_interleave(batch_num_of_nodes).reshape(-1, 1)
        return node_feats / norm_factors
