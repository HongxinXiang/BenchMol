from collections.abc import Sequence
from typing import cast

import pandas as pd
import torch
from base_classes import GeoGNNDataElement


def load_smiles_csv(
    csv_path: str,
    smiles_column_name: str,
    data_columns_to_load: list[str]
) -> list[GeoGNNDataElement]:
    """Loads a GeoGNN-dataset's CSV file. The CSV file should have a column
    containing the data-element's SMILES string, and one or more columns
    containing numerical data (eg. values for solubility, activation-energy, etc.).

    Args:
        csv_path (str): Path to the CSV file (ideally an absolute path, as \
            relative paths may cause in unexpected behavior).
        smiles_column_name (str): Column name/header of the SMILES string column.
        data_columns_to_load (list[str]): Column names/headers of the data to \
            be loaded.

    Returns:
        list[GeoGNNDataElement]: List of all loaded data from the CSV file, \
            with each element containing `"smiles"` (the element's SMILES string) \
            and `"data"` (tensor of the element's data from the columns specified \
            in `data_columns_to_load`).
    """
    raw_df = pd.read_csv(csv_path, sep=',')
    smiles_list = raw_df[smiles_column_name].values
    smiles_list = cast(Sequence[str], smiles_list)

    filtered_data = torch.tensor(raw_df[data_columns_to_load].values, dtype=torch.float32)

    data_list: list[GeoGNNDataElement] = []
    for i in range(len(filtered_data)):
        data_list.append({
            'smiles': smiles_list[i],
            'data': filtered_data[i]
        })
    return data_list
