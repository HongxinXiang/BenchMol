
import numpy as np
from base_classes import GeoGNNDataElement
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset, Subset


class ScaffoldSplitter:
    """
    Split dataset by Bemis-Murcko scaffolds using the `smiles` string in the
    data.

    Adapted from GeoGNN's `ScaffoldSplitter`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/splitters.py#L129-L206

    Which was adapted from:
    https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    """

    def split(
        self,
        dataset: Dataset[GeoGNNDataElement],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
    ) -> tuple[Subset[GeoGNNDataElement], Subset[GeoGNNDataElement], Subset[GeoGNNDataElement]]:
        """
        Args:
            dataset (Dataset[GeoGNNDataElement]): The dataset to split.
            frac_train (float): The fraction of data to be used for training.
            frac_valid (float): The fraction of data to be used for validation.
            frac_test (float): The fraction of data to be used for testing.

        Returns:
            tuple[Subset[GeoGNNDataElement], Subset[GeoGNNDataElement], Subset[GeoGNNDataElement]]: \
                The training, validation and testing subsets - `(train_dataset, valid_dataset, test_dataset)`.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset) # type: ignore

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i in range(N):
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(dataset[i]['smiles'], includeChirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx: list[int] = []
        valid_idx: list[int] = []
        test_idx: list[int] = []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)
        return train_dataset, valid_dataset, test_dataset
