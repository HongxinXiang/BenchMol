"""
Processing of esol dataset.

ESOL (delaney) is a standard regression data set,which is also called delaney dataset. In the dataset, you can find  the structure and water solubility data of 1128 compounds.  It's a good choice to validate machine learning models and to estimate solubility directly based on molecular structure which was encoded in SMILES string.

You can download the dataset from
https://moleculenet.org/datasets-1 and load it into pahelix reader creators.

This is a PyTorch equivalent of GeoGNN's `esol_dataset.py`:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/datasets/esol_dataset.py
"""

from typing import Literal, TypeAlias

from base_classes import GeoGNNDataset
from utils import abs_path

from .shared_definitions import load_smiles_csv

QM9_TASK_COL_NAMES: TypeAlias = Literal[
    'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', \
        'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom'
]

class QM9Dataset(GeoGNNDataset):
    """Geometric, energetic, electronic and thermodynamic properties of
    DFT-modelled small molecules.

    The `data` Tensor in each element contains the label-values for all the
    tasks specified by `task_column_name`, size `(len(task_column_name), )`.

    The dataset can be downloaded from:
    https://moleculenet.org/datasets-1

    ## Note:
    I couldn't find the official definition of each column in the CSV file, but
    DGL's `QM9EdgeDataset` [documentation](https://docs.dgl.ai/en/1.1.x/generated/dgl.data.QM9EdgeDataset.html)
    and DeepChem's `QM9 Datasets` [documentation](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#qm9-datasets)
    mentions the column definitions.
    """

    def __init__(
        self,
        task_column_name: list[QM9_TASK_COL_NAMES],
        csv_path: str = './chemrl_downstream_datasets/qm9/raw/qm9.csv',
    ) -> None:
        """
        Args:
            task_column_name (list[QM9_TASK_COL_NAMES]): List of all the column \
                names of the desired tasks.
            csv_path (str, optional): Path to the dataset's `.csv` file. \
                (relative paths will be relative to the file `QM9Dataset` is defined in) \
                Defaults to './chemrl_downstream_datasets/qm9/raw/qm9.csv'.
        """
        data_list = load_smiles_csv(
            smiles_column_name = 'smiles',
            data_columns_to_load = task_column_name, # type: ignore
            csv_path = abs_path(csv_path, __file__), # Set path relative to this file.
        )
        super().__init__(data_list)
