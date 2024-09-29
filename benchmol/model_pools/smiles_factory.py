import torch
from model_pools.smiles.molformer import Molformer
from model_pools.base_utils import get_predictor
from model_pools.smiles.chembert import Smiles_BERT, MolRoberta


class SmilesModelFactory(torch.nn.Module):
    def __init__(self, model_name, head_arch, num_tasks, vocab_path, d_dropout=0, head_arch_params=None, pretrain_path=None, device="cpu", **kwargs):
        super(SmilesModelFactory, self).__init__()

        self.model_name = model_name
        self.head_arch = head_arch
        self.num_tasks = num_tasks
        self.vocab_path = vocab_path
        self.d_dropout = d_dropout
        self.device = device

        if head_arch_params is None:
            head_arch_params = {"inner_dim": None, "dropout": 0.2, "activation_fn": None}
        self.head_arch_params = head_arch_params
        self.pretrain_path = pretrain_path

        self.head_predictor = None
        self.model = self.get_model()

    def forward(self, batch):
        features = self.model(batch)
        if self.head_predictor is None:
            return features
        else:
            return self.head_predictor(features)

    def get_model(self):
        if "molformer" in self.model_name:
            model = Molformer(n_embd=768, n_layer=6, n_head=12, num_feats=32, d_dropout=self.d_dropout, vocab_path=self.vocab_path, device=self.device)
            in_features = 768
        elif "CHEM-BERT" == self.model_name or "CHEM-BERT-RANDOM" == self.model_name:
            seq, nhead, embed_size, model_dim, layers, adjacency, drop_rate = 256, 16, 1024, 1024, 6, True, 0
            model = Smiles_BERT(max_len=seq, nhead=nhead, feature_dim=embed_size, feedforward_dim=model_dim,
                                nlayers=layers, adj=adjacency, dropout_rate=drop_rate, ret_feat=True,
                                task_type="regression", device=self.device)
            in_features = embed_size
        elif "CHEM-BERT-origin" == self.model_name or "CHEM-BERT-origin-RANDOM" == self.model_name:
            seq, nhead, embed_size, model_dim, layers, adjacency, drop_rate = 256, 16, 1024, 1024, 8, True, 0
            model = Smiles_BERT(max_len=seq, nhead=nhead, feature_dim=embed_size, feedforward_dim=model_dim,
                                nlayers=layers, adj=adjacency, dropout_rate=drop_rate, ret_feat=True,
                                task_type="regression", device=self.device)
            in_features = embed_size
        elif "CHEM-RoBERTa" in self.model_name:
            seq, nhead, embed_size, layers, adjacency, drop_rate = 256, 16, 768, 12, True, 0
            model = MolRoberta(max_len=seq, nhead=nhead, feature_dim=embed_size, nlayers=layers, adj=adjacency,
                               dropout_rate=drop_rate, ret_feat=True, task_type="regression", device=self.device)
            in_features = embed_size
        else:
            raise ValueError

        if self.pretrain_path != "None" and self.pretrain_path != "none" and self.pretrain_path is not None:
            print(f"load checkpoint from {self.pretrain_path}")
            model.from_pretrained(self.pretrain_path)

        # self-defined head for prediction
        if self.head_arch != "none" and self.head_arch != "None" and self.head_arch is not None:
            self.head_predictor = get_predictor(arch=self.head_arch, in_features=in_features, num_tasks=self.num_tasks,
                                                inner_dim=self.head_arch_params["inner_dim"],
                                                dropout=self.head_arch_params["dropout"],
                                                activation_fn=self.head_arch_params["activation_fn"])
        return model


if __name__ == '__main__':
    model = SmilesModelFactory(model_name="molformer", head_arch="none", num_tasks=5,
                               vocab_path="smiles/molformer_helper/tokenizer/bert_vocab.txt").model
    print(model)

