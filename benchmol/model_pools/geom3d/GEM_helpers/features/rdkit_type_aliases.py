"""
Aliases for RDKit's types, as RDKit doesn't have proper type hintings.
"""

from typing import Any, NewType

from rdkit.Chem import rdchem  # type: ignore

Atom = rdchem.Atom
"""A molecular atom (alias for `rdkit.Chem.rdchem.Atom`)."""

Bond = rdchem.Bond
"""A molecular bond (alias for `rdkit.Chem.rdchem.Bond`)."""

Mol = rdchem.Mol
"""A molecule (alias for `rdkit.Chem.rdchem.Mol`)."""

Conformer = rdchem.Conformer
"""A molecular conformer (alias for `rdkit.Chem.rdchem.Conformer`)."""

RDKitEnum = NewType('RDKitEnum', Any)
"""An enum from `rdkit.Chem.rdchem` (eg. `rdchem.ChiralType`)."""

RDKitEnumValue = NewType('RDKitEnumValue', Any)
"""A value for an enum in `rdkit.Chem.rdchem` (eg. `rdchem.ChiralType.CHI_UNSPECIFIED`)."""
