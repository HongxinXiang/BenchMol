from torch import Tensor, nn


class DropoutMLP(nn.Module):
    """
    Multi-layer perceptron (MLP) with dropout layers.

    This is a PyTorch equivalent of GeoGNN's `MLP`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/basic_block.py#L42-L68
    """

    def __init__(
        self,
        num_of_layers: int,
        in_size: int,
        hidden_size: int,
        out_size: int,
        activation: nn.Module,
        dropout_rate: float,
    ):
        """
        Args:
            num_of_layers (int): Number of layers. Should be > 1, else there \
                won't be any dropout layers.
            in_size (int): Input feature size.
            hidden_size (int): Hidden layers' size.
            out_size (int): Output feature size.
            activation (nn.Module): Activation module (eg. `nn.ReLU()`).
            dropout_rate (float): Dropout rate for the dropout layers.
        """
        super().__init__()
        assert num_of_layers > 1, "`num_of_layers` should be > 1, else there won't be any dropout layers."

        layers: list[nn.Module] = []
        for layer_id in range(num_of_layers):
            is_last_layer = layer_id == num_of_layers - 1
            if is_last_layer:
                layers.append(nn.Linear(hidden_size, out_size))
                continue

            is_first_layer = layer_id == 0
            layers.extend([
                nn.Linear(in_size if is_first_layer else hidden_size, hidden_size),
                nn.Dropout(dropout_rate),
                activation
            ])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input features of size `(in_size, )`, as defined in the constructor.

        Returns:
            Tensor: Output features of size `(out_size, )`, as defined in the constructor.
        """
        return self.mlp.forward(x)
