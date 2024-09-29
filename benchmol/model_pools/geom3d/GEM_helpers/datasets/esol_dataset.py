"""
Processing of esol dataset.

ESOL (delaney) is a standard regression data set,which is also called delaney dataset. In the dataset, you can find  the structure and water solubility data of 1128 compounds.  It's a good choice to validate machine learning models and to estimate solubility directly based on molecular structure which was encoded in SMILES string.

You can download the dataset from
https://moleculenet.org/datasets-1 and load it into pahelix reader creators.

This is a PyTorch equivalent of GeoGNN's `esol_dataset.py`:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/datasets/esol_dataset.py
"""

from base_classes import GeoGNNDataset
from utils import abs_path

from .shared_definitions import load_smiles_csv


class ESOLDataset(GeoGNNDataset):
    """
    Water solubility data (log-solubility in mols/L) for common small organic
    molecules.

    The `data` Tensor in each element is the ground truth water log-solublity
    in mols/L, size `(1, )`.

    The dataset can be downloaded from:
    https://moleculenet.org/datasets-1
    """

    def __init__(
        self,
        csv_path: str = './chemrl_downstream_datasets/esol/raw/delaney-processed.csv',
    ) -> None:
        """
        Args:
            csv_path (str, optional): Path to the dataset's `.csv` file. \
                Defaults to './chemrl_downstream_datasets/esol/raw/delaney-processed.csv'.
        """
        data_list = load_smiles_csv(
            smiles_column_name = 'smiles',
            data_columns_to_load = ['measured log solubility in mols per litre'],
            csv_path = abs_path(csv_path, __file__), # Set path relative to this file.
        )
        super().__init__(data_list)
