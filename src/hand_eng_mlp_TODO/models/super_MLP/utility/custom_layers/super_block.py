from typing import Type, Callable
import torch.nn as nn
from torch import Tensor
from src.hand_eng_mlp_TODO.models.super_MLP.utility.custom_layers.super_linear import SuperLinear


class SuperBlock(nn.Module):
    def __init__(
            self, in_features: int, mid_out: int, out_features: int,
            alpha: float = 0.5, f: Type[nn.Module] = nn.ReLU, dropout: float = .0
    ) -> None:

        super().__init__()

        if dropout > .0:
            self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        else:
            self.dropout: Callable[[Tensor], Tensor] = lambda x: x

        self.super_linear: SuperLinear = SuperLinear(
            in_features=in_features, out_features=mid_out, alpha=alpha
        )

        self.f: nn.Module = f()

        self.linear: nn.Linear = nn.Linear(
            in_features=mid_out, out_features=out_features
        )

    def forward(self, x_batch) -> Tensor:
        dropped_out: Tensor = self.dropout(x_batch)

        super_out: Tensor = self.super_linear(dropped_out)
        super_out = self.f(super_out)

        return self.linear(super_out)
