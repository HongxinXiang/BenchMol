"""
Features as defined in GeoGNN's utility functions/classes:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/compound_tools.py

Not all features in the above `compound_tools.py` are included, as not all
was actually used by the GeoGNN model. Only those specified in the GeoGNN's
config are included:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json
"""

from typing import Any, Final

import torch
from typing_extensions import Never

from .base_feature_classes import FloatFeature, LabelEncodedFeature


def raise_not_implemented(*_: Any) -> Never:
    raise NotImplementedError(
        "The bond-angle-feature values are computed in the `smiles_to_graphs` function." \
            + " It's not implemented here because computing them is closely tied to" \
            + " generating the bond-angle graph, and it'll be too much hassle to separate" \
            + " them. Also, there's only 1 bond-angle feature at the moment, so there's" \
            + " not much benefit to making the bond-angle-features code reusable."
    )

bond_angle = FloatFeature(
    name = 'bond_angle',

    # Centers and gamma values as defined in GeoGNN's `BondAngleFloatRBF` layer:
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L168
    rbf_centers = torch.arange(0, torch.pi, 0.1),
    rbf_gamma = 10.0,
    get_feat_values = raise_not_implemented,
)



# # This is unused and thus the GNNs are not configured to use it.
# # Hence, it's commented out to avoid confusing readers.
# LABEL_ENCODED_BOND_ANGLE_FEATURES: Final[list[LabelEncodedFeature]] = []
# """
# All predefined label-encoded bond-angle features that'll be in the graphs.
# """

FLOAT_BOND_ANGLE_FEATURES: Final[list[FloatFeature]] = [
    bond_angle,
]
"""
All predefined bond-angle features that have feature values of datatype `float`
that'll be in the graphs.
"""
