import math
from abc import ABC

import torch
from base_classes import GeoGNNCacheDataModule, GeoGNNDataElement, GeoGNNGraphs
from torch.utils.data import Dataset, random_split
from typing_extensions import override

from .datasets import QM9_TASK_COL_NAMES, QM9Dataset
from .preprocessing import smiles_to_graphs


class _BaseDataModule(GeoGNNCacheDataModule, ABC):
    """Base `LightningDataModule` class for molecular-properties prediction."""

    @override
    @classmethod
    def compute_graphs(cls, smiles: str) -> GeoGNNGraphs:
        return smiles_to_graphs(smiles, torch.device('cpu'))


class QM9DataModule(_BaseDataModule):
    """`LightningDataModule` class for QM9 dataset with random-split.

    DataGeometric, energetic, electronic and thermodynamic properties of
    DFT-modelled small molecules.

    The labels Tensor in are the tasks specified by `task_column_name` in the
    constructor.

    The dataset can be downloaded from:
    https://moleculenet.org/datasets-1

    ## Note:
    I couldn't find the official definition of each column in the CSV file, but
    DGL's `QM9EdgeDataset` [documentation](https://docs.dgl.ai/en/1.1.x/generated/dgl.data.QM9EdgeDataset.html)
    and DeepChem's `QM9 Datasets` [documentation](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#qm9-datasets)
    mentions the column definitions.
    """

    @override
    def get_dataset_splits(self) -> \
        tuple[Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement]]:
        dataset = QM9Dataset(self.task_column_name)

        # Based on `random_split` from newer versions of Pytorch:
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
        # But I couldn't install newer versions of Pytorch as they didn't support
        # CUDA 11.3 (which my machine was running on).
        train_val_test_frac = [0.8, 0.1, 0.1]
        subset_lengths: list[int] = []
        for frac in train_val_test_frac:
            n_items_in_split = int(math.floor(len(dataset) * frac))
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1

        rand_gen = torch.Generator().manual_seed(self.random_split_seed)
        train_dataset, valid_dataset, test_dataset \
            = random_split(dataset, subset_lengths, generator=rand_gen)
        return train_dataset, test_dataset, valid_dataset

    def __init__(
        self,
        task_column_name: list[QM9_TASK_COL_NAMES],
        batch_size: int,
        shuffle: bool = False,
        cache_path: str | None = None,
        random_split_seed: int = 0,
    ):
        """
        Args:
            task_column_name (list[QM9_TASK_COL_NAMES]): List of all the column \
                names of the desired tasks.
            batch_size (int): Batch size.
            shuffle (bool, optional): Whether to enable shuffling in the dataloaders. \
                Defaults to False.
            cache_path (str | None, optional): Path to existing graph-cache file, \
                or on where to generate a new cache file should the it not exist. \
                If `None`, the graphs are computed when they're getted from the \
                dataset (instead of all being precomputed in `self.setup`). \
                Defaults to None.
            random_split_seed (int, optional): Seed for performing random-split on \
                the dataset. Defaults to 0.
        """
        self.task_column_name = task_column_name
        self.random_split_seed = random_split_seed
        super().__init__(batch_size, shuffle, cache_path)
