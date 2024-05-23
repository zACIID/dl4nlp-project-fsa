from torch import tensordot, Tensor
import torch.nn as nn
import torch as t


def find_split(multi: int, target: int) -> int:
    while target % multi:
        multi += 1

    return multi


class SuperLinear(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            alpha: float = 0.5, n_dim: int = 1
    ) -> None:
        super().__init__()

        reduction_layer_out: int = find_split(int(out_features * alpha), out_features)
        self.reduction_layer: nn.Linear = nn.Linear(
            in_features=in_features, out_features=reduction_layer_out
        )

        self.alpha: float = reduction_layer_out / out_features

        amplify_layer_dim: int = out_features // reduction_layer_out
        std_dev: float = 1 / out_features
        self.V: nn.Parameter = nn.Parameter(data=t.randn(amplify_layer_dim) * std_dev)
        self.b_V: nn.Parameter = nn.Parameter(data=t.randn(out_features) * std_dev)

        self.n_dim: int = n_dim
        self.in_features: int = in_features
        self.out_features: int = out_features

    def forward(self, x_batch: Tensor) -> Tensor:
        reduction_out: Tensor = self.reduction_layer(x_batch)

        tensor_prod: Tensor = tensordot(reduction_out, self.V, dims=0)
        dim: int = 1 if self.n_dim == x_batch.dim() else x_batch.shape[0]

        if dim > 1:
            tensor_prod = tensor_prod.view(dim, self.out_features)
        else:
            tensor_prod = tensor_prod.view(self.out_features)

        return tensor_prod + self.b_V
