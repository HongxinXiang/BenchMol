import glob
import re

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset


class Vocab(object):
    def __init__(self):
        self.pad_index = 0
        self.mask_index = 1
        self.unk_index = 2
        self.start_index = 3
        self.end_index = 4

        # check 'Na' later
        self.voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', '(',
                                                                             'n', '2', 'c', 'F', ')', '=', 'N', '3',
                                                                             'S', '/', 's', '-', '+', 'o', 'P', 'R',
                                                                             '\\', 'L', '#', 'X', '6', 'B', '7', '4',
                                                                             'I', '5', 'i', 'p', '8', '9', '%', '0',
                                                                             '.', ':', 'A']
        self.dict = {s: i for i, s in enumerate(self.voca_list)}

    def __len__(self):
        return len(self.voca_list)


class ChemBertDataset(Dataset):
    def __init__(self, df, vocab, seq_len, mat_position, label_column="label", task_type="classification"):
        self.vocab = vocab
        self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
        self.smiles_dataset = []
        self.adj_dataset = []
        self.mat_pos = mat_position
        self.seq_len = seq_len

        smiles_list = df["smiles"].tolist()
        labels = np.array(df[label_column].apply(lambda x: str(x).split(' ')).tolist())
        labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
        self.labels = labels.reshape((len(labels), -1))
        self.df = df

        for i in smiles_list:
            self.adj_dataset.append(i)
            self.smiles_dataset.append(self.replace_halogen(i))

    def __len__(self):
        return len(self.smiles_dataset)

    def __getitem__(self, idx):
        # print(idx)
        item = self.smiles_dataset[idx]
        label = self.labels[idx]

        input_token, input_adj_masking = self.CharToNum(item)

        input_data = [self.vocab.start_index] + input_token + [self.vocab.end_index]
        input_adj_masking = [0] + input_adj_masking + [0]
        if self.mat_pos == 'start':
            input_adj_masking = [1] + [0 for _ in range(len(input_adj_masking) - 1)]

        smiles_bert_input = input_data[:self.seq_len]
        smiles_bert_adj_mask = input_adj_masking[:self.seq_len]

        padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
        smiles_bert_input.extend(padding)
        smiles_bert_adj_mask.extend(padding)

        mol = Chem.MolFromSmiles(self.adj_dataset[idx])
        # features = add_descriptors(mol)
        # smiles_bert_ECFP = np.array(features, dtype=np.float32)
        if mol != None:
            adj_mat = GetAdjacencyMatrix(mol)
            smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))
        else:
            smiles_bert_adjmat = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)

        output = {"smiles_bert_input": smiles_bert_input, "label": label,
                  "smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat}

        return {key: torch.tensor(value) for key, value in output.items()}

    def CharToNum(self, smiles):
        tokens = [i for i in smiles]
        adj_masking = []

        for i, token in enumerate(tokens):
            if token in self.atom_vocab:
                adj_masking.append(1)
            else:
                adj_masking.append(0)

            tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

        return tokens, adj_masking

    def replace_halogen(self, string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        sn = re.compile('Sn')
        na = re.compile('Na')
        string = br.sub('R', string)
        string = cl.sub('L', string)
        string = sn.sub('X', string)
        string = na.sub('A', string)
        return string

    def zero_padding(self, array, shape):
        if array.shape[0] > shape[0]:
            array = array[:shape[0], :shape[1]]
        padded = np.zeros(shape, dtype=np.float32)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded
