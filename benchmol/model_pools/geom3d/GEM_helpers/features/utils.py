from .rdkit_type_aliases import RDKitEnum, RDKitEnumValue


def _rdkit_enum_to_list(rdkit_enum: RDKitEnum) -> list[RDKitEnumValue]:
    """
    Converts an enum from `rdkit.Chem.rdchem` (eg. `rdkit.Chem.rdchem.ChiralType`)
    to a list of all the possible enum values
    (eg. `[ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ...]`)

    Args:
        rdkit_enum (RDKitEnum): An enum from `rdkit.Chem.rdchem`.

    Returns:
        list[RDKitEnumValue]: All possible enum values in a list.
    """
    return [rdkit_enum.values[i] for i in range(len(rdkit_enum.values))]
