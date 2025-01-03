import os

import torch

from benchmol.model_pools.graph.deepergcn import load_DeeperGCN


class CGIP_G(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.gnn = load_DeeperGCN(num_layers=14, hidden_channels=512, dropout=dropout)
        self.graph_pred_linear = None

    def forward(self, batch):
        graph_features = self.gnn(batch)
        return self.graph_pred_linear(graph_features)

    def from_pretrained(self, pretrain_gnn_path, model_key="model_state_dict2", consistency=False, logger=None):
        log = logger.info if logger is not None else print
        flag = False  # load successfully when only flag is true
        desc = None
        if pretrain_gnn_path:
            if os.path.isfile(pretrain_gnn_path):
                log("===> Loading checkpoint '{}'".format(pretrain_gnn_path))
                checkpoint = torch.load(pretrain_gnn_path)

                # load parameters
                ckpt_model_state_dict = checkpoint[model_key]
                if consistency:  # model and ckpt_model_state_dict is consistent.
                    self.gnn.load_state_dict(ckpt_model_state_dict)
                    log("load all the parameters of pre-trianed model.")
                else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                    ckp_keys = list(ckpt_model_state_dict)
                    cur_keys = list(self.gnn.state_dict())
                    len_ckp_keys = len(ckp_keys)
                    len_cur_keys = len(cur_keys)
                    model_sd = self.gnn.state_dict()
                    for idx in range(min(len_ckp_keys, len_cur_keys)):
                        ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
                        # print(ckp_key, cur_key)
                        model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                    self.gnn.load_state_dict(model_sd)
                    log("load the first {} parameters. layer number: model({}), pretrain({})"
                             .format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))

                desc = "[resume model info] The pretrained_model is at checkpoint {}. \t Loss value: {}" \
                    .format(checkpoint['epoch'], checkpoint["loss"])
                log(desc)
                flag = True
            else:
                log("===> No checkpoint found at '{}'".format(pretrain_gnn_path))
        else:
            log('===> No pre-trained model')
        return flag, desc
