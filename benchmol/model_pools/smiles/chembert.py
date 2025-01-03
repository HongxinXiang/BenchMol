import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.models.roberta import RobertaModel, RobertaConfig

from benchmol.model_pools.smiles.abc_smiles import SmilesABC
from benchmol.model_pools.smiles.chembert_helper.dataset import Vocab, ChemBertDataset
from benchmol.model_pools.smiles.chembert_helper.model_component import Smiles_embedding
from torch.utils.data import default_collate


# Reference: https://github.com/HyunSeobKim/CHEM-BERT


class Smiles_BERT(nn.Module):  # , SmilesABC
    def __init__(self, max_len=256, feature_dim=1024, nhead=4, feedforward_dim=1024, nlayers=6, adj=False, dropout_rate=0, ret_feat=False, task_type="classification", device="cpu"):
        super(Smiles_BERT, self).__init__()
        self.max_len = max_len
        self.Smiles_vocab = Vocab()
        self.vocab_size = len(self.Smiles_vocab)
        self.adj = adj
        self.ret_feat = ret_feat
        self.task_type = task_type
        self.device = device

        self.embedding = Smiles_embedding(self.vocab_size, feature_dim, max_len, adj=adj)
        trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu',
                                                 dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)

    # self.linear = Masked_prediction(feedforward_dim, vocab_size)

    def forward(self, batch):
        position_num = torch.arange(self.max_len).repeat(batch["smiles_bert_input"].size(0), 1).to(self.device)
        if self.adj:
            src, pos_num, adj_mask, adj_mat = batch["smiles_bert_input"], position_num, batch["smiles_bert_adj_mask"], batch["smiles_bert_adjmat"]
        else:
            src, pos_num, adj_mask, adj_mat = batch["smiles_bert_input"], position_num, None, None
        # True -> masking on zero-padding. False -> do nothing
        # mask = (src == 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
        mask = (src == 0)
        mask = mask.type(torch.bool)
        # print(mask.shape)

        x = self.embedding(src, pos_num, adj_mask, adj_mat)
        x = self.transformer_encoder(x.transpose(1, 0), src_key_padding_mask=mask)
        x = x.transpose(1, 0)

        if self.ret_feat:
            return x[:, 0]
        else:
            return x

    def from_pretrained(self, pretrain_path, model_key=None, consistency=True, logger=None):
        # print(f"load checkpoint from {pretrain_path}")
        self.load_state_dict(torch.load(pretrain_path))

    def get_dataset(self, csv_path, task_type, seq_len=256, mat_position="atom", label_column="label"):
        df = pd.read_csv(csv_path)
        dataset = ChemBertDataset(df, self.Smiles_vocab, seq_len=seq_len, mat_position="atom", label_column=label_column, task_type=self.task_type)
        return dataset

    def collate(self, batch):
        return default_collate(batch)


class MolRoberta(nn.Module):  # , SmilesABC
    def __init__(self, max_len=256, feature_dim=768, nhead=12, max_position_embeddings=512, nlayers=12, adj=False, dropout_rate=0, ret_feat=False, task_type="classification", device="cpu"):
        super(MolRoberta, self).__init__()
        self.max_len = max_len
        self.Smiles_vocab = Vocab()
        self.vocab_size = len(self.Smiles_vocab)
        self.adj = adj
        self.ret_feat = ret_feat
        self.task_type = task_type
        self.device = device

        self.embedding = Smiles_embedding(self.vocab_size, feature_dim, max_len, adj=adj)

        config = RobertaConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=max_position_embeddings + 2,  # must add 2
            num_attention_heads=nhead,
            num_hidden_layers=nlayers,
            hidden_size=feature_dim,
            type_vocab_size=1,  # 一个样本中输入的句子个数
            dropout_rate=dropout_rate
        )
        self.config = config
        self.transformer_encoder = RobertaModel(config=config).encoder

    def forward(self, batch):
        position_num = torch.arange(self.max_len).repeat(batch["smiles_bert_input"].size(0), 1).to(self.device)
        if self.adj:
            src, pos_num, adj_mask, adj_mat = batch["smiles_bert_input"], position_num, batch["smiles_bert_adj_mask"], batch["smiles_bert_adjmat"]
        else:
            src, pos_num, adj_mask, adj_mat = batch["smiles_bert_input"], position_num, None, None
        # True -> masking on zero-padding. False -> do nothing
        # mask = (src == 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
        mask = (src == 0)
        mask = mask.type(torch.bool)
        # print(mask.shape)

        x = self.embedding(src, pos_num, adj_mask, adj_mat)
        x = self.transformer_encoder(x, torch.unsqueeze(torch.unsqueeze(mask, 1), 1))
        last_hidden_state = x.last_hidden_state

        if self.ret_feat:
            return last_hidden_state[:, 0]
        else:
            return last_hidden_state

    def from_pretrained(self, pretrain_path, model_key=None, consistency=True, logger=None):
        # print(f"load checkpoint from {pretrain_path}")
        self.load_state_dict(torch.load(pretrain_path))

    def get_dataset(self, csv_path, task_type, seq_len=256, mat_position="atom", label_column="label"):
        df = pd.read_csv(csv_path)
        Smiles_vocab = Vocab()
        dataset = ChemBertDataset(df, Smiles_vocab, seq_len=seq_len, mat_position="atom", label_column=label_column, task_type=self.task_type)
        return dataset

    def collate(self, batch):
        return default_collate(batch)


if __name__ == '__main__':
    model_name = "Smiles_BERT_origin"
    if model_name == "Smiles_BERT":
        seq, nhead, embed_size, model_dim, layers, adjacency, drop_rate = 256, 16, 1024, 1024, 6, True, 0
        model = Smiles_BERT(max_len=seq, nhead=nhead, feature_dim=embed_size, feedforward_dim=model_dim, nlayers=layers, adj=adjacency, dropout_rate=drop_rate, ret_feat=True)
        model.from_pretrained(pretrain_path="../../checkpoints/pretrained-smiles/CHEM-BERT.pt")
    elif model_name == "Smiles_BERT_origin":
        seq, nhead, embed_size, model_dim, layers, adjacency, drop_rate = 256, 16, 1024, 1024, 8, True, 0
        model = Smiles_BERT(max_len=seq, nhead=nhead, feature_dim=embed_size, feedforward_dim=model_dim, nlayers=layers, adj=adjacency, dropout_rate=drop_rate, ret_feat=True)
        model.from_pretrained(pretrain_path="../../checkpoints/pretrained-smiles/CHEM-BERT-origin.pt")
    elif model_name == "MolRoberta":
        seq, nhead, embed_size, layers, adjacency, drop_rate = 256, 16, 768, 12, True, 0
        model = MolRoberta(max_len=seq, nhead=nhead, feature_dim=embed_size, nlayers=layers, adj=adjacency, dropout_rate=drop_rate, ret_feat=True)
        model.from_pretrained(pretrain_path="../../checkpoints/pretrained-smiles/CHEM-RoBERTa.pt")
    else:
        raise Exception

    dataset = model.get_dataset(csv_path="../../datasets/bbbp_processed_ac.csv", task_type="classification")
    dataloader = DataLoader(dataset, batch_size=16, num_workers=2, shuffle=False, collate_fn=model.collate)

    model.eval()
    for batch in dataloader:
        feat = model(batch)
        print(feat.shape)
