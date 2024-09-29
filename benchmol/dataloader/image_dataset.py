import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader.data_utils import get_labels_from_df, read_image


def get_image_path_list(root, data_type, index_list, img_dir="", return_dataset_class=False):
    image_path_list = []
    if data_type == "image":
        for index in index_list:
            image_path_list.append(f"{root}/image_2d_rdkit/{index}.png")
        dataset_class = ImageDataset
    elif data_type == "image_2":  # 第二种格式
        for index in index_list:
            image_path_list.append(f"{root}/image/{index}/{index}.png")
        dataset_class = ImageDataset
    elif data_type == "multi_view_image":
        assert isinstance(img_dir, str)
        for index in index_list:
            image_path_list.append([f"{root}/{img_dir}/{index}/x_0.png", f"{root}/{img_dir}/{index}/x_180.png",
                                    f"{root}/{img_dir}/{index}/y_180.png", f"{root}/{img_dir}/{index}/z_180.png"])
        dataset_class = MultiViewImageDataset
    elif data_type == "multi_conf_multi_view_image":
        assert isinstance(img_dir, list)
        for conf_dir in img_dir:
            temp_list = []
            for index in index_list:
                temp_list.append(
                    [f"{root}/{conf_dir}/{index}/x_0.png", f"{root}/{conf_dir}/{index}/x_180.png",
                     f"{root}/{conf_dir}/{index}/y_180.png", f"{root}/{conf_dir}/{index}/z_180.png"])
            image_path_list.append(temp_list)
        dataset_class = MultiConfImageDataset
    elif data_type == "video":
        assert isinstance(img_dir, str)
        for index in index_list:
            video_path_list = [f"{root}/{img_dir}/{index}/x_{idx}.png" for idx in range(0, 360, 18)] + \
                              [f"{root}/{img_dir}/{index}/y_{idx}.png" for idx in range(0, 360, 18)] + \
                              [f"{root}/{img_dir}/{index}/z_{idx}.png" for idx in range(0, 360, 18)]
            image_path_list.append(video_path_list)
        dataset_class = MultiViewImageDataset
    else:
        raise ValueError

    if return_dataset_class:
        return image_path_list, dataset_class

    return image_path_list


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, transform, img_type="RGB"):
        '''
        :param names: image path, shape: (n_sample, ) e.g. ["./data/1.png", "./data/2.png", ..., "./data/n.png"]
        :param labels: labels, e.g. single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer:
        :param args:
        '''

        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.transform = transform
        self.img_type = img_type

    def get_image(self, index):
        filename = self.filenames[index]
        image = self.transform(read_image(filename, img_type=self.img_type))
        return image

    def __getitem__(self, index):
        return {
            "images": self.get_image(index),
            "labels": self.labels[index]
        }

    def __len__(self):
        return self.total


class MultiViewImageDataset(Dataset):
    def __init__(self, filenames, labels, transform, img_type="RGB"):
        '''
        :param names: image path, shape: (n_sample, n_view) e.g. [["./data/1.png"], ["./data/2.png"], ..., ["./data/n.png"]]
        :param labels: labels, e.g. single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer:
        :param args:
        '''
        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.transform = transform
        self.img_type = img_type

    def get_image(self, index):
        filenames = self.filenames[index]
        images = [self.transform(read_image(filename, img_type=self.img_type)) for filename in filenames]
        return torch.stack(images, dim=0)

    def __getitem__(self, index):
        return {
            "images": self.get_image(index),
            "labels": self.labels[index]
        }

    def __len__(self):
        return self.total


class MultiConfImageDataset(Dataset):
    def __init__(self, filenames, labels, transform, img_type="RGB"):
        '''
        :param names: image path, shape: (n_conf, n_sample, n_view)
        :param labels: labels, e.g. single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer:
        :param args:
        '''
        filenames = np.array(filenames)
        self.n_conf, self.total, self.n_view = filenames.shape
        self.filenames = filenames.transpose(1, 0, 2)
        self.labels = labels
        self.transform = transform
        self.img_type = img_type

    def get_image(self, index):
        filenames = self.filenames[index]
        multi_conf_images = []
        for idx_conf in range(self.n_conf):
            multi_conf_images.append(torch.stack([self.transform(read_image(filename, img_type=self.img_type)) for filename in filenames[idx_conf]], dim=0))
        return torch.stack(multi_conf_images, dim=0)

    def __getitem__(self, index):
        return {
            "images": self.get_image(index),
            "labels": self.labels[index]
        }

    def __len__(self):
        return self.total


class TrainValTestFromCSVFactory():
    """从 CSV 文件中构建 train, valid, test 的 Dataset
    csv 中必须有一列提供分割，分割字段分别是：train, valid, test
    """
    def __init__(self, root, csv_path, data_type, image_dir_name, task_type, y_column="label", split_column="scaffold_split",
                 batch_size=8, num_workers=2, normalize=None, transform=None, eval_transform=None, pin_memory=False, logger=None):
        self.root = root
        self.csv_path = csv_path
        self.data_type = data_type
        self.image_dir_name = image_dir_name
        self.task_type = task_type
        self.df = pd.read_csv(self.csv_path)
        self.split_column = split_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.log = logger.info if logger is not None else print

        self.index = self.df["index"]
        self.labels = get_labels_from_df(self.df, task_type, y_column=y_column)
        self.num_tasks = self.labels.shape[-1]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]) if normalize is None else normalize
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                             self.normalize]) if transform is None else transform
        self.eval_transform = self.transform if eval_transform is None else eval_transform

    def get_image_path_list(self, root, data_type, index_list, image_dir_name, return_dataset_class=False):
        image_path_list = []
        if data_type == "image":
            for index in index_list:
                image_path_list.append(f"{root}/{image_dir_name}/{index}.png")
            dataset_class = ImageDataset
        elif data_type == "image_2":  # 第二种格式
            for index in index_list:
                image_path_list.append(f"{root}/{image_dir_name}/{index}/{index}.png")
            dataset_class = ImageDataset
        elif data_type == "multi_view_image":
            for index in index_list:
                image_path_list.append([f"{root}/{image_dir_name}/{index}/x_0.png", f"{root}/{image_dir_name}/{index}/x_180.png",
                                        f"{root}/{image_dir_name}/{index}/y_180.png", f"{root}/{image_dir_name}/{index}/z_180.png"])
            dataset_class = MultiViewImageDataset

        elif data_type == "multi_view_image_2":  # 第二种命名方式
            for index in index_list:
                image_path_list.append([f"{root}/{image_dir_name}/{index}/0.png", f"{root}/{image_dir_name}/{index}/10.png",
                                        f"{root}/{image_dir_name}/{index}/30.png", f"{root}/{image_dir_name}/{index}/50.png"])
            dataset_class = MultiViewImageDataset

        elif data_type == "multi_conf_multi_view_image":
            for image_dir in image_dir_name.split(","):
                temp_list = []
                for index in index_list:
                    temp_list.append(
                        [f"{root}/{image_dir}/{index}/x_0.png", f"{root}/{image_dir}/{index}/x_180.png",
                         f"{root}/{image_dir}/{index}/y_180.png", f"{root}/{image_dir}/{index}/z_180.png"])
                image_path_list.append(temp_list)
            dataset_class = MultiConfImageDataset

        elif data_type == "video":
            for index in index_list:
                video_path_list = [f"{root}/{image_dir_name}/{index}/x_{idx}.png" for idx in range(0, 360, 18)] + \
                                  [f"{root}/{image_dir_name}/{index}/y_{idx}.png" for idx in range(0, 360, 18)] + \
                                  [f"{root}/{image_dir_name}/{index}/z_{idx}.png" for idx in range(0, 360, 18)]
                image_path_list.append(video_path_list)
            dataset_class = MultiViewImageDataset

        elif data_type == "video_2":
            for index in index_list:
                video_path_list = [f"{root}/{image_dir_name}/{index}/{idx}.png" for idx in range(0, 60)]
                image_path_list.append(video_path_list)
            dataset_class = MultiViewImageDataset

        else:
            raise ValueError

        if return_dataset_class:
            return image_path_list, dataset_class

        return image_path_list

    def get_dataloader(self, split):
        assert split in ["train", "valid", "test"]

        idx_split = self.df[self.split_column] == split

        split_image_path_list, dataset_class = self.get_image_path_list(root=self.root, data_type=self.data_type,
                                                                        index_list=self.index[idx_split].tolist(),
                                                                        image_dir_name=self.image_dir_name,
                                                                        return_dataset_class=True)

        if split == "train":
            transform = self.transform
            self.log(f"[split={split}] use transform")
            self.log(f"[train] split_image_path_list[0]: {split_image_path_list[0]}")
        else:
            self.log(f"[split={split}] use eval_transform")
            transform = self.eval_transform

        split_labels = self.labels[idx_split]
        shuffle = True if split == "train" else False

        dataset = dataset_class(split_image_path_list, labels=split_labels, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=shuffle)

        return dataloader