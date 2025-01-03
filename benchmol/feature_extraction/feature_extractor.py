import abc
import dataclasses
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from benchmol.dataloader.image_dataset import ImageDataset, MultiViewImageDataset, MultiConfImageDataset
from benchmol.dataloader.graph_dataset import GraphDataset
from benchmol.dataloader.collater import Collater
from benchmol.data_process.molecules.get_fp_features import FeaturesGeneration
from benchmol.data_process.molecules.fingerprints import fpFunc_dict
from sklearn.preprocessing import Normalizer


@dataclasses.dataclass
class FeatureExtractor(abc.ABC):

    @abc.abstractmethod
    def extract_features(self):
        pass

    @abc.abstractmethod
    def return_features(self):
        pass


class FingerprintExtractor(FeatureExtractor):

    def __init__(self, smiles_list, fp_names):  # fp_name 支持 ',' 进行指纹拼接
        self.smiles_list = smiles_list
        self.fp_names = fp_names.split(",")
        for fp_name in self.fp_names:
            assert fp_name in FingerprintExtractor.get_fp_names()
        self.featuresGeneration = FeaturesGeneration()

    def extract_features(self, norm=False):
        features = []
        for fp_name in self.fp_names:
            feature = self.featuresGeneration.get_fingerprints_from_smiles_list(smiles_list=self.smiles_list,
                                                                                fp_name=fp_name)

            if norm:
                transformer = Normalizer().fit(feature)
                feature = transformer.transform(feature)

            features.append(feature)
        self.features = np.hstack(features)
        # print(f"extract fingerprint {self.fp_names}: shape: {self.features.shape}, "
        #       f"mean: {np.mean(self.features)}, std: {np.std(self.features)}")

    def return_features(self, norm=False):
        if self.features is None:
            warnings.warn(f"fingerprint is not cached, generating new fingerprint with norm={norm}")
            self.extract_features(norm)
        return self.features

    @staticmethod
    def get_fp_names():
        return list(fpFunc_dict.keys())

class SmilesFeatureExtractor(FeatureExtractor):

    def __init__(self, model, csv_path, batch_size=8, num_workers=2, pin_memory=False, device="cpu"):
        self.model = model
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.features = None

    def extract_features(self):
        dataset = self.model.get_dataset(self.csv_path, task_type="regression")  # task_type 这个参数没用，为了不必要的麻烦，设置成 regression
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, collate_fn=self.model.collate)
        feats = []
        self.model.eval()
        for batch in tqdm(dataloader, desc=f"extract features"):
            if isinstance(batch, dict):
                batch = {item: batch[item].to(self.device) for item in batch.keys()}
            else:
                batch = batch.to(self.device)
            feat = self.model(batch)
            feats.append(feat.detach().cpu().numpy())
        self.features = np.concatenate(feats, axis=0)

    def return_features(self):
        if self.features is None:
            self.extract_features()
        return self.features


class GraphFeatureExtractor(FeatureExtractor):

    def __init__(self, model, graph_path, graph_feat=None,
                 batch_size=8, num_workers=2, pin_memory=False, device="cpu"):
        self.model = model
        self.graph_path = graph_path
        self.graph_data, self.graph_slices = torch.load(graph_path)
        if graph_feat == "edge_eq_2":  # edge 只使用前 2 个维度的特征
            self.graph_data.edge_attr = self.graph_data.edge_attr[:, :2]
        elif graph_feat == "min":  # 和 pretrain-gnns 配置一样
            self.graph_data.edge_attr = self.graph_data.edge_attr[:, :2]
            self.graph_data.x = self.graph_data.x[:, :2]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.features = None

    def extract_features(self):
        dataset = GraphDataset(data=self.graph_data, slices=self.graph_slices)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, collate_fn=Collater(follow_batch=[], multigpu=False))
        feats = []
        self.model.eval()
        for batch in tqdm(dataloader, desc=f"extract features"):
            batch = batch.to(self.device)
            feat = self.model(batch)
            feats.append(feat.detach().cpu().numpy())
        self.features = np.concatenate(feats, axis=0)

    def return_features(self):
        if self.features is None:
            self.extract_features()
        return self.features


class ImageFeatureExtractor(FeatureExtractor):
    def __init__(self, model, filenames,
                 normalize=None, transform=None, batch_size=8, num_workers=2, pin_memory=False, device="cpu", img_type="RGB"):
        self.model = model
        self.filenames = filenames
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]) if normalize is None else normalize
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                             self.normalize]) if transform is None else transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.features = None
        self.img_type = img_type

    def extract_features(self):
        dataset = ImageDataset(self.filenames, labels=[-1]*len(self.filenames), transform=self.transform, img_type=self.img_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        self.model.eval()
        feats = []
        for data_package in tqdm(dataloader, desc=f"extract features"):
            x = data_package["images"]
            bs, c, h, w = x.shape
            x = x.to(self.device)
            feat = self.model(x)
            feats.append(feat.detach().cpu().numpy())
        self.features = np.concatenate(feats, axis=0)

    def return_features(self):
        if self.features is None:
            self.extract_features()
        return self.features

    def __len__(self):
        return len(self.filenames)


class MVImageFeatureExtractor(FeatureExtractor):
    """Supporting Multi-view Image
    filenames: (n_sample, n_view)
    """
    def __init__(self, model, filenames, pooling_view=True,
                 normalize=None, transform=None, batch_size=8, num_workers=2, pin_memory=False, device="cpu", img_type="RGB"):
        self.model = model
        self.filenames = filenames
        self.pooling_view = pooling_view
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]) if normalize is None else normalize
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                             self.normalize]) if transform is None else transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.features = None
        self.img_type = img_type

    def extract_features(self):
        dataset = MultiViewImageDataset(self.filenames, labels=[-1]*len(self.filenames), transform=self.transform, img_type=self.img_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        self.model.eval()
        feats = []
        for data_package in tqdm(dataloader, desc=f"extract features"):
            x = data_package["images"]
            bs, n_view, c, h, w = x.shape
            x = x.reshape(bs * n_view, c, h, w).to(self.device)
            if self.pooling_view:
                feat = self.model(x).reshape(bs, n_view, -1).mean(1)
            else:
                feat = self.model(x).reshape(bs, n_view, -1)
            feats.append(feat.detach().cpu().numpy())
        self.features = np.concatenate(feats, axis=0)

    def return_features(self):
        if self.features is None:
            self.extract_features()
        return self.features

    def __len__(self):
        return len(self.filenames)


class MCImageFeatureExtractor(FeatureExtractor):
    """Supporting Multi-view Image with Multi-Conformations
    filenames: (n_sample, n_conf, n_view)
    """
    def __init__(self, model, filenames, pooling_conf=True, pooling_view=True,
                 normalize=None, transform=None, batch_size=8, num_workers=2, pin_memory=False, device="cpu", img_type="RGB"):
        self.n_conf, self.total, self.n_view = np.array(filenames).shape
        self.model = model
        self.filenames = filenames
        self.pooling_conf = pooling_conf
        self.pooling_view = pooling_view
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]) if normalize is None else normalize
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                             self.normalize]) if transform is None else transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.features = None
        self.img_type = img_type

    def extract_features(self):
        dataset = MultiConfImageDataset(self.filenames, labels=[-1]*self.total, transform=self.transform, img_type=self.img_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        self.model.eval()
        feats = []
        for data_package in tqdm(dataloader, desc=f"extract features"):
            x = data_package["images"]
            bs, n_conf, n_view, c, h, w = x.shape
            x = x.reshape(bs * n_conf * n_view, c, h, w).to(self.device)
            feat = self.model(x).reshape(bs, n_conf, n_view, -1)
            feat = feat.mean(2) if self.pooling_view else feat
            feat = feat.mean(1) if self.pooling_conf else feat
            feats.append(feat.detach().cpu().numpy())
        self.features = np.concatenate(feats, axis=0)

    def return_features(self):
        if self.features is None:
            self.extract_features()
        return self.features

    def __len__(self):
        return self.total

