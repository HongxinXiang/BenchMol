from typing import Any, Literal, overload

from base_classes import GeoGNNLightningModule, LoggedHyperParams
from dgl import DGLGraph
from torch import Tensor, nn
from typing_extensions import override

from .encoder_model import GeoGNNModel
from .layers import DropoutMLP


class DownstreamModel(GeoGNNLightningModule):
    """Downstream model for molecular-property prediction.

    Uses the graph-representation output from `self.encoder: GeoGNNModel` to
    make `out_size` number of predictions.

    This is a PyTorch + DGL equivalent of GeoGNN's `DownstreamModel`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/src/model.py#L27-L61
    """

    @overload
    def __init__(self, *, encoder: GeoGNNModel, task_type: Literal['classification', 'regression'], out_size: int, num_of_mlp_layers: int, mlp_hidden_size: int = 128, activation: nn.Module = nn.LeakyReLU(), dropout_rate: float = 0.2, lr: float = 1e-4, _logged_hparams: LoggedHyperParams = {}):
        """
        Default values for `hidden_size`, `activation` and `dropout_rate` are
        based on GeoGNN's `down_mlp2.json` / `down_mlp3.json` config:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/down_mlp2.json
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/down_mlp3.json

        where `down_mlp2.json` and `down_mlp3.json` are used by the GeoGNN's
        finetuning scripts:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/finetune_class.sh#L37
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/finetune_regr.sh#L37

        Both of GeoGNN's finetuning scripts tests both `num_of_layers = 2` and
        `num_of_layers = 3`.

        Args:
            encoder (GeoGNNModel): GeoGNN encoder for the molecule graphs.
            task_type (Literal['classification', 'regression']): Whether to \
                perform a classification or regression.
            out_size (int): Size of output tensor.
            num_of_mlp_layers (int): Number of layers in the dropout MLP.
            mlp_hidden_size (int, optional): Hidden size of dropout MLP. Defaults to 128.
            activation (nn.Module, optional): Activation layer to use. Defaults to `nn.LeakyReLU()`.
            dropout_rate (float, optional): Dropout rate of the dropout MLP. Defaults to 0.2.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            _logged_hparams (LoggedHyperParams, optional): Hyperparameters that's \
                not used by the model, but is logged in the lightning-log's \
                `hparams.yaml` file.. Defaults to {}.
        """

    @overload
    def __init__(self, *, encoder_params: dict[str, Any], task_type: Literal['classification', 'regression'], out_size: int, num_of_mlp_layers: int, mlp_hidden_size: int = 128, activation: nn.Module = nn.LeakyReLU(), dropout_rate: float = 0.2, lr: float = 1e-4, _logged_hparams: LoggedHyperParams = {}):
        """
        Args:
            encoder_params (dict[str, Any]): Params to init a new `GeoGNNModel` \
                instance with.
            task_type (Literal['classification', 'regression']): Whether to \
                perform a classification or regression.
            out_size (int): Size of output tensor.
            num_of_mlp_layers (int): Number of layers in the dropout MLP.
            mlp_hidden_size (int, optional): Hidden size of dropout MLP. Defaults to 128.
            activation (nn.Module, optional): Activation layer to use. Defaults to `nn.LeakyReLU()`.
            dropout_rate (float, optional): Dropout rate of the dropout MLP. Defaults to 0.2.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            _logged_hparams (LoggedHyperParams, optional): Hyperparameters that's \
                not used by the model, but is logged in the lightning-log's \
                `hparams.yaml` file.. Defaults to {}.
        """

    def __init__(
        self,
        *,
        encoder: GeoGNNModel | None = None,
        encoder_params: dict[str, Any] | None = None,
        task_type: Literal['classification', 'regression'],
        out_size: int,
        num_of_mlp_layers: int,
        mlp_hidden_size: int = 128,
        activation: nn.Module = nn.LeakyReLU(),
        dropout_rate: float = 0.2,
        lr: float = 1e-4,
        _logged_hparams: LoggedHyperParams = {},
    ):
        super().__init__(out_size, lr, _logged_hparams)
        assert (encoder != None and encoder_params == None) \
            or (encoder == None and encoder_params != None), \
            'Either `encoder` or `encoder_params` must be given, but not both. ' \
                + "I don't want to deal with the case where if both are given, " \
                + "and `encoder_params` doesn't match the params in `encoder`"

        self.encoder = encoder or GeoGNNModel(**encoder_params) # type: ignore
        encoder_params = {
            'embed_dim': self.encoder.embed_dim,
            'dropout_rate': self.encoder.dropout_rate,
            'num_of_layers': self.encoder.num_of_layers,
        }
        self.save_hyperparameters(ignore=['encoder'])

        self.task_type = task_type
        self.out_size = out_size

        self.norm = nn.LayerNorm(self.encoder.embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = num_of_mlp_layers,
            in_size = self.encoder.embed_dim,
            hidden_size = mlp_hidden_size,
            out_size = out_size,
            activation = activation,
            dropout_rate = dropout_rate,
        )
        if self.task_type == 'classification':
            self.out_act = nn.Sigmoid()

    @override
    def forward(self, batched_atom_bond_graph: DGLGraph, batched_bond_angle_graph: DGLGraph) -> Tensor:
        """
        Args:
            batched_atom_bond_graph (DGLGraph): Graph (or batched graph) of \
                molecules with atoms as nodes, bonds as edges.
            batched_bond_angle_graph (DGLGraph): Graph (or batched graph) of \
                molecules with bonds as nodes, bond-angles as edges.

        Returns:
            Tensor: Predicted values with size `(self.out_size, )`.
        """
        node_repr, edge_repr, graph_repr \
            = self.encoder.forward(batched_atom_bond_graph, batched_bond_angle_graph)
        graph_repr = self.norm.forward(graph_repr)
        pred = self.mlp.forward(graph_repr)

        if self.task_type == 'classification':
            return self.out_act.forward(pred)
        return pred
