from torch_geometric.data import InMemoryDataset
import torch
from torch.utils.data import DataLoader
from benchmol.dataloader.collater import Collater
import pandas as pd
import numpy as np
from benchmol.model_pools.smiles.molformer import MultitaskEmbeddingDataset
from benchmol.model_pools.smiles.chembert import Vocab, ChemBertDataset


class TrainValTestFromCSVFactory():
    """从 CSV 文件中构建 train, valid, test 的 Dataset
    csv 中必须有一列提供分割，分割字段分别是：train, valid, test
    """
    def __init__(self, csv_path, model_name, task_type, y_column="label", split_column="scaffold_split",
                 batch_size=8, num_workers=2, seq_len=100, pin_memory=False, collate_fn=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        self.model_name = model_name
        self.task_type = task_type
        self.y_column = y_column
        self.split_column = split_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if collate_fn is None:
            self.collate_fn = Collater(follow_batch=[], multigpu=False)
        else:
            self.collate_fn = collate_fn

    def get_dataloader(self, split, ):
        assert split in ["train", "valid", "test"]
        index = list(self.df[self.df[self.split_column] == split].index)
        assert len(np.unique(self.df.iloc[index][self.split_column])) == 1

        split_df = self.df.iloc[index].reset_index()

        if self.model_name == "molformer":
            split_dataset = MultitaskEmbeddingDataset(split_df, self.y_column, self.task_type)
        elif self.model_name in ["CHEM-BERT", "CHEM-BERT-origin", "CHEM-RoBERTa"]:
            split_dataset = ChemBertDataset(split_df, Vocab(), seq_len=256, mat_position="atom", label_column=self.y_column, task_type=self.task_type)
        else:
            raise NotImplementedError

        shuffle = True if split == "train" else False
        dataloader = DataLoader(split_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, collate_fn=self.collate_fn)
        return dataloader