from dataclasses import dataclass
from typing import Any, Callable

import torch
from dgl import DGLGraph
from torch import FloatTensor, IntTensor, Tensor

from .rdkit_type_aliases import Atom, Bond, Conformer, Mol


@dataclass
class Feature:
    """Base class for all features."""

    name: str
    """Name that'll be used as the keys for `DGLGraph.ndata` and `DGLGraph.edata`."""

    get_feat_values: Callable[[Mol, Conformer, DGLGraph], Tensor]
    """Get the feature's values from rdkit's `Mol` and `Conformer` instances
    (and optionally from the current `DGLGraph` instance, in the case where this
    feature depends on another feature present in the graph, eg. temp feats.).
    """


@dataclass
class FloatFeature(Feature):
    # Overridden fields from `Feature` class.
    name: str
    """Name that'll be used as the keys for `DGLGraph.ndata` and `DGLGraph.edata`."""
    get_feat_values: Callable[[Mol, Conformer, DGLGraph], FloatTensor]
    """Get the feature's values from rdkit's `Mol` and `Conformer` instances
    (and optionally from the current `DGLGraph` instance, in the case where this
    feature depends on another feature present in the graph, eg. temp feats.).
    """


    # New fields.
    rbf_centers: Tensor
    """1D tensor of all the RBF centers for a feature, used in `FixedRBF`.

    In the original GeoGNN code, these values are defined in the constructor of
    `BondFloatRBF` and `BondAngleFloatRBF` in:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py
    """

    rbf_gamma: float
    """Hyperparameter for controlling the spread of the RBF's Gaussian
    basis-function, used in `FixedRBF`.

    In the original GeoGNN code, these values are defined in the constructor of
    `BondFloatRBF` and `BondAngleFloatRBF` in:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py
    """

    @staticmethod
    def create_atom_feat(
        name: str,
        rbf_centers: Tensor,
        rbf_gamma: float,
        get_value: Callable[[Atom, Mol, Conformer, DGLGraph], float],
        dtype: torch.dtype = torch.float32,
    ) -> 'FloatFeature':
        """Factory method for creating an atom feature from a function
        (ie. `get_value`) that gets the feat's value from a single rdkit's `Atom`
        instance.
        """
        return FloatFeature(
            rbf_centers = rbf_centers,
            rbf_gamma = rbf_gamma,
            name = name,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                get_value(atom, mol, conf, atom_bond_graph) for atom in mol.GetAtoms()
            ], dtype=dtype),
        )

    @staticmethod
    def create_bond_feat(
        name: str,
        rbf_centers: Tensor,
        rbf_gamma: float,
        get_value: Callable[[Bond, Mol, Conformer, DGLGraph], float],
        dtype: torch.dtype = torch.float32,
    ) -> 'FloatFeature':
        """Factory method for creating a bond feature from a function
        (ie. `get_value`) that gets the feat's value from a single rdkit's `Bond`
        instance.
        """
        return FloatFeature(
            rbf_centers = rbf_centers,
            rbf_gamma = rbf_gamma,
            name = name,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                get_value(bond, mol, conf, atom_bond_graph) for bond in mol.GetBonds()
            ], dtype=dtype),
        )


@dataclass
class LabelEncodedFeature(Feature):
    # Overridden fields from `Feature` class.
    name: str
    """Name that'll be used as the keys for `DGLGraph.ndata` and `DGLGraph.edata`."""
    get_feat_values: Callable[[Mol, Conformer, DGLGraph], IntTensor]
    """Get the feature's values from rdkit's `Mol` and `Conformer` instances
    (and optionally from the current `DGLGraph` instance, in the case where this
    feature depends on another feature present in the graph, eg. temp feats.).
    """


    # New fields.
    possible_values: list[Any]

    @staticmethod
    def create_atom_feat(
        name: str,
        possible_values: list[Any],
        get_raw_value: Callable[[Atom, Mol, Conformer, DGLGraph], Any],
        dtype: torch.dtype = torch.int64,
    ) -> 'LabelEncodedFeature':
        """Factory method for creating an atom feature from a function
        (ie. `get_raw_value`) that gets the feat's actual (ie. raw) feature
        value from a single rdkit's `Atom` instance. This "raw" value is then
        label encoded into an `int`.
        """
        return LabelEncodedFeature(
            name = name,
            possible_values = possible_values,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                LabelEncodedFeature._encoded_value(
                    raw_value = get_raw_value(atom, mol, conf, atom_bond_graph),
                    possible_values = possible_values,
                ) for atom in mol.GetAtoms()
            ], dtype=dtype),
        )

    @staticmethod
    def create_bond_feat(
        name: str,
        possible_values: list[Any],
        get_raw_value: Callable[[Bond, Mol, Conformer, DGLGraph], Any],
        dtype: torch.dtype = torch.int64,
    ) -> 'LabelEncodedFeature':
        """Factory method for creating a bond feature from a function
        (ie. `get_raw_value`) that gets the feat's actual (ie. raw) feature
        value from a single rdkit's `Bond` instance. This "raw" value is then
        label encoded into an `int`.
        """
        return LabelEncodedFeature(
            name = name,
            possible_values = possible_values,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                LabelEncodedFeature._encoded_value(
                    raw_value = get_raw_value(bond, mol, conf, atom_bond_graph),
                    possible_values = possible_values,
                ) for bond in mol.GetBonds()
            ], dtype=dtype),
        )

    @staticmethod
    def _encoded_value(
        raw_value: Any,
        possible_values: list[Any],
    ) -> int:
        """Label-encode the feature's actual (ie. raw) value.

        Args:
            raw_value (Any): Actual feature value \
                (eg. `rdkit.Chem.rdchem.HybridizationType.SP2` enum object).
            possible_values (list[Any]): List of all the possible raw values that \
                this feature can take (eg. list of the enums in \
                `rdkit.Chem.rdchem.HybridizationType.values`).

        Returns:
            int: Label-encoded value, 1-indexed with reference to the `possible_values` \
                list (`0` is reserved for the absence of the feature).
        """
        try:
            # 1-indexed, as `0` is reserved for the absence of the feature.
            return possible_values.index(raw_value) + 1
        except ValueError:
            # If `value` doesn't exist in `self.possible_values`,
            # return the last element's index (assumes 'misc' is the last index).
            return len(possible_values) - 1
