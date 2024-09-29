
import dgl, torch
from dgl import DGLGraph
from torch import Tensor


def to_bidirected_copy(g: DGLGraph) -> DGLGraph:
    """Make a bidirected copy of the uni-directed graph `g`, adding a new
    opposing directed edge for each edge in `g` and copying that edge's
    features.

    The new opposing edges are interleaved with the original edges.
    (eg. `[edge1, edge1_opp, edge2, edge2_opp, ...]`)

    ### Warning:
    Reuses node feat tensors of `g` (instead of copying them).

    Args:
        g (DGLGraph): The input directed graph.

    Returns:
        DGLGraph: Graph `g` but bidirected and with copied node/edge features.
    """
    # Get the current edges and their reverse.
    src, dst = g.edges()
    rev_src, rev_dst = dst, src

    # Interleave the original and reversed edges.
    interleaved_src = torch.empty(len(src) * 2).to(src)
    interleaved_dst = torch.empty(len(dst) * 2).to(dst)
    interleaved_src[::2] = src
    interleaved_src[1::2] = rev_src
    interleaved_dst[::2] = dst
    interleaved_dst[1::2] = rev_dst

    # Create a new graph with interleaved edges.
    bidirected_g = dgl.graph((interleaved_src, interleaved_dst), num_nodes=g.num_nodes())

    # Copy the node features.
    for feat_name, feat_tensor in g.ndata.items():
        assert isinstance(feat_name, str) and isinstance(feat_tensor, Tensor)
        bidirected_g.ndata[feat_name] = feat_tensor

    # Copy and interleave the edge features.
    for feat_name, feat_tensor in g.edata.items():
        assert isinstance(feat_name, str) and isinstance(feat_tensor, Tensor)
        bidirected_g.edata[feat_name] = torch.repeat_interleave(feat_tensor, 2, dim=0)

    return bidirected_g
