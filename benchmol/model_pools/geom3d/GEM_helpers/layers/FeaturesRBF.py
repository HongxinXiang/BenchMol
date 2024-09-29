from typing import Mapping, cast

import torch
from torch import Tensor, nn

from ..features import FloatFeature
from .FixedRBF import FixedRBF


class FeaturesRBF(nn.Module):
    """
    Transforms multiple features (eg. node/edge features) into a tensor of size
    `(num_elements, self.output_dim)`.

    It works by first converting each feature into a tensor of size
    `(num_elements, self.output_dim)`, then summing all the features' tensors
    into one tensor.

    This is a generalised PyTorch equivalent of GeoGNN's `BondFloatRBF` and
    `BondAngleFloatRBF`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L121-L192
    """

    def __init__(
        self,
        float_feat_list: list[FloatFeature],
        output_dim: int,
    ):
        """
        Args:
            float_feat_list (list[FloatFeature]): Info on the float-features.
            output_dim (int): The output embedding dimension.
        """
        super().__init__()
        self.output_dim = output_dim
        self.module_dict = nn.ModuleDict()
        for feat in float_feat_list:
            layer = nn.Sequential(
                FixedRBF(feat.rbf_centers, feat.rbf_gamma),
                nn.Linear(len(feat.rbf_centers), output_dim),
            )
            self.module_dict[feat.name] = layer

    def reset_parameters(self) -> None:
        """
        Resets the weights of all the `nn.Linear` submodules by calling
        `nn.Linear.reset_parameters()` on each of them.
        """
        for _, sequential in self.module_dict.items():
            assert isinstance(sequential, nn.Sequential)
            cast(nn.Linear, sequential[1]).reset_parameters()

    def forward(self, feat_tensor_map: Mapping[str, Tensor]) -> Tensor:
        """
        Args:
            feat_tensor_map (Mapping[str, Tensor]): Mapping of \
                features, where the keys are the features' names and the values \
                are the features' `Tensor` values (eg. from `DGLGraph.ndata`).

        Returns:
            Tensor: Embedding of size `(num_elements, self.output_dim)`.
        """
        feat_values = list(feat_tensor_map.values())
        num_of_elements = len(feat_values[0])
        device = next(self.parameters()).device
        output_embed = torch.zeros(num_of_elements, self.output_dim, dtype=torch.float32, device=device)

        for feat_name, tensor in feat_tensor_map.items():
            if feat_name not in self.module_dict:
                continue
            layer = self.module_dict[feat_name]
            assert isinstance(layer, nn.Sequential)

            layer_output = layer.forward(tensor)
            assert isinstance(layer_output, Tensor)

            output_embed += layer_output

        return output_embed
