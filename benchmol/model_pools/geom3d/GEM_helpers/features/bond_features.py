"""
Features as defined in GeoGNN's utility functions/classes:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/compound_tools.py

Not all features in the above `compound_tools.py` are included, as not all
was actually used by the GeoGNN model. Only those specified in the GeoGNN's
config are included:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json
"""

from typing import Final

import torch
from dgl import DGLGraph
from rdkit.Chem import rdchem  # type: ignore
from torch import Tensor

from .base_feature_classes import Feature, FloatFeature, LabelEncodedFeature
from .rdkit_type_aliases import Conformer, Mol
from .utils import _rdkit_enum_to_list

bond_dir = LabelEncodedFeature.create_bond_feat(
    name = 'bond_dir',
    possible_values = _rdkit_enum_to_list(rdchem.BondDir),
    get_raw_value = lambda x, *_ : x.GetBondDir(),
)


bond_type = LabelEncodedFeature.create_bond_feat(
    name = 'bond_type',
    possible_values = _rdkit_enum_to_list(rdchem.BondType),
    get_raw_value = lambda x, *_ : x.GetBondType(),
)


is_in_ring = LabelEncodedFeature.create_bond_feat(
    name = 'is_in_ring',
    possible_values = [0, 1],
    get_raw_value = lambda x, *_ : int(x.IsInRing()),
)


def _bond_length_get_feat_values(mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph):
    """`get_feat_values` Callable function for `bond_length` feature."""
    assert atom_bond_graph != None, 'Bond length feature requires `atom_bond_graph`.'
    assert '_atom_pos' in atom_bond_graph.ndata, \
        "Bond length feat requires 3D atom position feat to be first computed at `atom_bond_graph.ndata['_atom_pos']`."

    atom_positions = atom_bond_graph.ndata['_atom_pos']
    assert isinstance(atom_positions, Tensor)

    edges_tuple: tuple[Tensor, Tensor] = atom_bond_graph.edges()
    src_node_idx, dst_node_idx = edges_tuple

    # To use as tensor indexing, these index tensors needs to be `dtype=long`.
    src_node_idx, dst_node_idx = src_node_idx.long(), dst_node_idx.long()

    return torch.norm(atom_positions[dst_node_idx] - atom_positions[src_node_idx], dim=1)

bond_length = FloatFeature(
    name = 'bond_length',

    # Centers and gamma values as defined in GeoGNN's `BondFloatRBF` layer:
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L131
    rbf_centers = torch.arange(0, 2, 0.1),
    rbf_gamma = 10.0,
    get_feat_values = _bond_length_get_feat_values,
)



LABEL_ENCODED_BOND_FEATURES: Final[list[LabelEncodedFeature]] = [
    bond_dir,
    bond_type,
    is_in_ring,
]
"""
All predefined label-encoded bond features that'll be in the graphs.
"""

FLOAT_BOND_FEATURES: Final[list[FloatFeature]] = [
    bond_length,
]
"""
All predefined bond features that have feature values of datatype `float`
that'll be in the graphs.
"""
