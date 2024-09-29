import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from fast_transformers.masking import LengthMask as LM
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader

from model_pools.smiles.abc_smiles import SmilesABC
from model_pools.smiles.molformer_helper.rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from model_pools.smiles.molformer_helper.tokenizer.tokenizer import MolTranBertTokenizer
from dataloader.data_utils import get_labels_from_df

# Reference: https://github.com/IBM/molformer

def normalize_smiles(smi, canonical, isomeric):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


class MultitaskEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, df, label_column="label", task_type="classification"):
        if "canonical_smiles" not in df.columns:
            df['canonical_smiles'] = df['smiles'].apply(lambda smi: normalize_smiles(smi, canonical=True, isomeric=False))
        self.labels = get_labels_from_df(df, task_type, label_column)
        self.num_tasks = self.labels.shape[-1]
        self.df = df

    def __getitem__(self, index):
        canonical_smiles = self.df.loc[index, 'canonical_smiles']
        measures = self.labels[index]
        mask = [0.0 if np.isnan(x) else 1.0 for x in measures]
        measures = [0.0 if np.isnan(x) else x for x in measures]
        return canonical_smiles, measures, mask

    def __len__(self):
        return len(self.df)


class Molformer(torch.nn.Module):  # SmilesABC
    def __init__(self, n_embd, n_layer, n_head, num_feats, d_dropout, vocab_path, device="cpu"):
        super(Molformer, self).__init__()

        tokenizer = MolTranBertTokenizer(vocab_path)
        self.tokenizer = tokenizer

        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=n_layer,
            n_heads=n_head,
            query_dimensions=n_embd//n_head,
            value_dimensions=n_embd//n_head,
            feed_forward_dimensions=n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.drop = nn.Dropout(d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(n_embd, n_vocab)

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def from_pretrained(self, pretrain_path, model_key="state_dict", consistency=True, logger=None):
        log = logger.info if logger is not None else print
        flag = False  # load successfully when only flag is true
        desc = None
        if pretrain_path:
            if os.path.isfile(pretrain_path):
                log("===> Loading checkpoint '{}'".format(pretrain_path))
                checkpoint = torch.load(pretrain_path)

                # load parameters
                ckpt_model_state_dict = checkpoint[model_key]
                if consistency:  # model and ckpt_model_state_dict is consistent.
                    self.load_state_dict(ckpt_model_state_dict)
                    log("load all the parameters of pre-trianed model.")
                else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                    ckp_keys = list(ckpt_model_state_dict)
                    cur_keys = list(self.state_dict())
                    len_ckp_keys = len(ckp_keys)
                    len_cur_keys = len(cur_keys)
                    model_sd = self.state_dict()
                    for idx in range(min(len_ckp_keys, len_cur_keys)):
                        ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
                        # print(ckp_key, cur_key)
                        model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                    self.load_state_dict(model_sd)
                    log("load the first {} parameters. layer number: model({}), pretrain({})".format(
                        min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))

                desc = "[resume model info] The pretrained_model is at checkpoint {}.".format(checkpoint['epoch'])
                log(desc)
                flag = True
            else:
                log("===> No checkpoint found at '{}'".format(pretrain_path))
        else:
            log('===> No pre-trained model')
        return flag, desc

    def collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus([smile[0] for smile in batch], padding=True, add_special_tokens=True)
        return {
            "input_ids": torch.tensor(tokens['input_ids']),
            "attention_mask": torch.tensor(tokens['attention_mask']),
            "label": torch.tensor([smile[1] for smile in batch], dtype=torch.float32),
            "label_mask": torch.tensor([smile[2] for smile in batch])
        }

    def get_dataset(self, csv_path, label_column="label", task_type="classification"):
        df = pd.read_csv(csv_path)
        dataset = MultitaskEmbeddingDataset(df, label_column, task_type)
        return dataset

    def forward(self, batch):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

        b, t = input_ids.size()
        # forward the GPT model
        token_embeddings = self.tok_emb(input_ids)  # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(attention_mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask

        return loss_input


if __name__ == '__main__':
    vocab_path = './molformer_helper/tokenizer/bert_vocab.txt'
    molformer = Molformer(n_embd=768, n_layer=6, n_head=12, num_feats=32, d_dropout=0, vocab_path=vocab_path)
    molformer.from_pretrained(pretrain_path="../../checkpoints/pretrained-smiles/molformer.ckpt")
    print(molformer)

    dataset = molformer.get_dataset(csv_path="../../datasets/bbbp_processed_ac.csv", task_type="classification")
    dataloader = DataLoader(dataset, batch_size=16, num_workers=2, shuffle=False, collate_fn=molformer.collate)

    molformer.eval()
    for batch in dataloader:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        label, label_mask = batch["label"], batch["label_mask"]

        feat = molformer(batch)

        print(feat.shape)
