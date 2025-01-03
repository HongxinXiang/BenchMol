import torch

from benchmol.model_pools.base_utils import get_predictor
from benchmol.model_pools.graph.gnns import GNN_graphpred as basic_GNN_1
from benchmol.model_pools.graph.gnns_molebert import GNN_graphpred as basic_GNN_2
from benchmol.model_pools.graph.cgip import CGIP_G


class GraphModelFactory(torch.nn.Module):
    def __init__(self, model_name, head_arch, num_tasks, head_arch_params=None, pretrain_gnn_path=None,
                 model_key=None, num_layer=5, emb_dim=300, JK="last", dropout=0.5, graph_pooling="mean", gnn_type="gin",
                 update_predictor=True, **kwargs):
        super(GraphModelFactory, self).__init__()

        self.model_name = model_name
        self.head_arch = head_arch
        self.num_tasks = num_tasks
        if head_arch_params is None:
            head_arch_params = {"inner_dim": None, "dropout": 0.2, "activation_fn": None}
        self.head_arch_params = head_arch_params
        self.pretrain_gnn_path = pretrain_gnn_path
        self.model_key = model_key

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.graph_pooling = graph_pooling

        self.update_predictor = update_predictor
        self.model = self.get_model(update_predictor)

    def forward(self, batch):
        return self.model(batch)

    def get_model(self, update_predictor=True):
        if self.model_name in ["CGIP_G", "CGIP_G_RANDOM"]:
            model = CGIP_G()
            in_features = 512
        else:
            if self.model_name in [""]:
                model = basic_GNN_1(num_layer=self.num_layer, emb_dim=self.emb_dim, num_tasks=self.num_tasks, JK=self.JK,
                                    drop_ratio=self.dropout, graph_pooling=self.graph_pooling, gnn_type=self.gnn_type)
                self.mult = model.mult
            else:
                model = basic_GNN_2(num_layer=self.num_layer, emb_dim=self.emb_dim, num_tasks=self.num_tasks, JK=self.JK,
                                 drop_ratio=self.dropout, graph_pooling=self.graph_pooling, gnn_type=self.gnn_type)
                self.mult = model.mult

            if self.JK == "concat":
                in_features = self.mult * (self.num_layer + 1) * self.emb_dim
            else:
                in_features = self.mult * self.emb_dim

        if self.pretrain_gnn_path != "None" and self.pretrain_gnn_path is not None:
            print(f"load checkpoint from {self.pretrain_gnn_path}")
            model.from_pretrained(self.pretrain_gnn_path)

        # self-defined head for prediction
        head_predictor = get_predictor(arch=self.head_arch, in_features=in_features, num_tasks=self.num_tasks,
                                       inner_dim=self.head_arch_params["inner_dim"],
                                       dropout=self.head_arch_params["dropout"],
                                       activation_fn=self.head_arch_params["activation_fn"])
        head_predictor.requires_grad_(requires_grad=update_predictor)

        model.graph_pred_linear = head_predictor
        return model


if __name__ == '__main__':
    model = GraphModelFactory(model_name="MoleBERT", head_arch="arch1", num_tasks=5).model
    print(model)

