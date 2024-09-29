from torch_geometric.data import InMemoryDataset
import torch
from torch.utils.data import DataLoader
from dataloader.collater import Collater
import pandas as pd
import numpy as np


class GeometryDataset(InMemoryDataset):
    def __init__(self, data, slices, transform=None, pre_transform=None, pre_filter=None, task_type="classification"):
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super().__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = data, slices
        self.num_tasks = self.data.y.shape[1]
        self.total = len(self)
        self.task_type = task_type


class TrainValTestFromCSVFactory():
    """从 CSV 文件中构建 train, valid, test 的 Dataset
    csv 中必须有一列提供分割，分割字段分别是：train, valid, test
    """
    def __init__(self, csv_path, geometry_path, split_column="scaffold_split", geometry_feat=None,
                 batch_size=8, num_workers=2, pin_memory=False, collate_fn=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        self.split_column = split_column
        self.geometry_path = geometry_path
        self.geometry_data, self.geometry_slices = torch.load(geometry_path)
        if geometry_feat == "edge_eq_2":  # edge 只使用前 2 个维度的特征
            self.geometry_data.edge_attr = self.geometry_data.edge_attr[:, :2]
        elif geometry_feat == "min":  # 和 pretrain-gnns 配置一样
            self.geometry_data.edge_attr = self.geometry_data.edge_attr[:, :2]
            self.geometry_data.x = self.geometry_data.x[:, :2]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if collate_fn is None:
            self.collate_fn = Collater(follow_batch=[], multigpu=False)
        else:
            self.collate_fn = collate_fn

        assert len(self.geometry_data.y) == self.df.shape[0]
        self.dataset = GeometryDataset(data=self.geometry_data, slices=self.geometry_slices)
        self.num_tasks = self.dataset.num_tasks

    def get_dataloader(self, split):
        assert split in ["train", "valid", "test"]
        index = list(self.df[self.df[self.split_column] == split].index)
        split_dataset = self.dataset[index]
        shuffle = True if split == "train" else False
        dataloader = DataLoader(split_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, collate_fn=self.collate_fn)
        return dataloader

    def get_all_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, collate_fn=self.collate_fn)
        return dataloader