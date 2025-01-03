import collections.abc as container_abcs

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from benchmol.utils.splitter import *

from benchmol.model_pools.geom3d.UniMol.unimol_tools.data.conformer import coords2unimol
from benchmol.model_pools.geom3d.UniMol.unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords

string_classes, int_classes = str, int


class Collater(object):
    def __init__(self, follow_batch, multigpu=False):
        self.follow_batch = follow_batch
        self.multigpu = multigpu

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            if self.multigpu:
                return batch
            else:
                return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, np.ndarray):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class UniMolCollater(object):
    def __init__(self, dictionary, follow_batch, multigpu=False, max_atoms=256):
        self.dictionary = dictionary
        self.follow_batch = follow_batch
        self.multigpu = multigpu
        self.max_atoms = max_atoms

    def collate(self, batch):
        data_dict = [coords2unimol(item.atoms, item.coordinates, self.dictionary, max_atoms=self.max_atoms) for item in batch]

        index = torch.tensor([item.index for item in batch])
        # smi = [item.smi for item in batch]
        # scaffold = [item.scaffold for item in batch]
        y = torch.from_numpy(np.vstack([item.y for item in batch]))

        new_data_dict = {
            "src_coord": [torch.tensor(d["src_coord"]).float() for d in data_dict],
            "src_edge_type": [torch.tensor(d["src_edge_type"]).long() for d in data_dict],
            "src_distance": [torch.tensor(d["src_distance"]).float() for d in data_dict],
            "src_tokens": [torch.tensor(d["src_tokens"]).long() for d in data_dict],
        }

        data = Data()
        for k in new_data_dict.keys():
            if k == 'src_coord':
                data.src_coord = pad_coords(new_data_dict[k], pad_idx=0.0)
            elif k == 'src_edge_type':
                data.src_edge_type = pad_2d(new_data_dict[k], pad_idx=self.dictionary.pad())
            elif k == 'src_distance':
                data.src_distance = pad_2d(new_data_dict[k], pad_idx=0.0)
            elif k == 'src_tokens':
                data.src_tokens = pad_1d_tokens(new_data_dict[k], pad_idx=self.dictionary.pad())

        data.index = index
        data.y = y
        return data

    def __call__(self, batch):
        return self.collate(batch)


