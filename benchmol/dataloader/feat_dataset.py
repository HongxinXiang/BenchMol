import pickle

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataloader.data_utils import get_labels_from_df


class FeatDataset(Dataset):
    def __init__(self, feats, labels, ret_dict=False):
        assert len(feats) == len(labels)
        self.input_dim = len(feats[0])
        self.feats = feats
        self.labels = labels
        self.ret_dict = ret_dict
        self.total = len(self.feats)

    def __getitem__(self, index):
        if not self.ret_dict:
            return self.feats[index], self.labels[index]
        else:
            return {"feats": self.feats[index], "labels": self.labels[index]}
    def __len__(self):
        return self.total


class TrainValTestFromCSVFactory():
    """从 CSV 文件中构建 train, valid, test 的 Dataset
    csv 中必须有一列提供分割，分割字段分别是：train, valid, test
    """
    def __init__(self, csv_path, feat_pkl_path, task_type, y_column="label", split_column="scaffold_split",
                 batch_size=8, num_workers=8, pin_memory=False, logger=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

        with open(feat_pkl_path, "rb") as f:
            feat_dict = pickle.load(f)
        index, self.features, labels = feat_dict["index"], feat_dict["x"], feat_dict["y"]
        assert self.df["index"].tolist() == index and self.df[y_column].tolist() == labels.tolist()

        self.task_type = task_type
        self.split_column = split_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.log = logger.info if logger is not None else print

        self.index = self.df["index"]
        self.labels = get_labels_from_df(self.df, task_type, y_column=y_column)
        self.num_tasks = self.labels.shape[-1]

    def get_dataloader(self, split):
        assert split in ["train", "valid", "test"]

        idx_split = list(self.df[self.df[self.split_column] == split].index)
        split_feats, split_labels = self.features[idx_split], self.labels[idx_split]
        shuffle = True if split == "train" else False

        dataset = FeatDataset(feats=split_feats, labels=split_labels, ret_dict=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return dataloader
