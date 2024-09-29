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
from rdkit.Chem import rdchem  # type: ignore

from .base_feature_classes import Feature, FloatFeature, LabelEncodedFeature
from .utils import _rdkit_enum_to_list

atomic_num = LabelEncodedFeature.create_atom_feat(
    name = 'atomic_num',
    possible_values = list(range(1, 119)) + ['misc'],
    get_raw_value = lambda x, *_ : x.GetAtomicNum(),
)

chiral_tag = LabelEncodedFeature.create_atom_feat(
    name = 'chiral_tag',
    possible_values = _rdkit_enum_to_list(rdchem.ChiralType),
    get_raw_value = lambda x, *_ : x.GetChiralTag(),
)

degree = LabelEncodedFeature.create_atom_feat(
    name = 'degree',
    possible_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    get_raw_value = lambda x, *_ : x.GetDegree(),
)

formal_charge = LabelEncodedFeature.create_atom_feat(
    name = 'formal_charge',
    possible_values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    get_raw_value = lambda x, *_ : x.GetFormalCharge(),
)

hybridization = LabelEncodedFeature.create_atom_feat(
    name = 'hybridization',
    possible_values = _rdkit_enum_to_list(rdchem.HybridizationType),
    get_raw_value = lambda x, *_ : x.GetHybridization(),
)

is_aromatic = LabelEncodedFeature.create_atom_feat(
    name = 'is_aromatic',
    possible_values = [0, 1],
    get_raw_value = lambda x, *_ : x.GetIsAromatic(),
)

total_numHs = LabelEncodedFeature.create_atom_feat(
    name = 'total_numHs',
    possible_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    get_raw_value = lambda x, *_ : x.GetTotalNumHs(),
)

atom_pos = Feature(
    name = '_atom_pos',
    get_feat_values = lambda _, conf, *__ : torch.from_numpy(conf.GetPositions()).float(),
)



LABEL_ENCODED_ATOM_FEATURES: Final[list[LabelEncodedFeature]] = [
    atomic_num,
    chiral_tag,
    degree,
    formal_charge,
    hybridization,
    is_aromatic,
    total_numHs,
]
"""
All predefined label-encoded atom features that'll be in the graphs.
"""

# # This is unused and thus the GNNs are not configured to use it.
# # Hence, it's commented out to avoid confusing readers.
# FLOAT_ATOM_FEATURES: Final[list[FloatFeature]] = []
# """
# All predefined atom features that have feature values of datatype `float`
# that'll be in the graphs.
# """
