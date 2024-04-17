from enum import Enum

import torch as th
import torch.nn as nn


class HeadType(Enum):
    W_q = 1
    W_k = 2
    W_v = 3
    W_o = 4


# TODO at the end we might not need the bundle because we probably won't ever swap LoRAs
#   maybe remove?
class LoRABundle:
    def __init__(
            self,
            A: nn.Parameter,
            B: nn.Parameter,
            bias: nn.Parameter,
            layer: int,
            alpha: float,
            head_type: HeadType
    ) -> None:
        self.A = A
        self.B = B
        self.bias = bias
        self.layer = layer
        self.alpha = alpha
        self.head_type = head_type


class CustomLoRA(nn.Module):

    def __init__(
            self,
            old_linear: nn.Linear,
            layer: int = None,
            head_type: HeadType = None,
            rank: int = 1,
            alpha: float = 1,
            update_bias: bool = True
    ) -> None:

        super().__init__()

        self.old_linear: nn.Linear = old_linear
        self.old_bias: nn.Parameter = nn.Parameter(self.old_linear.bias.detach(), requires_grad=False)

        self.old_linear.weight.requires_grad = False
        if update_bias:
            self.old_linear.bias.requires_grad = True
        else:
            self.old_linear.bias.requires_grad = False

        std_dev: float = 1 / th.sqrt(th.tensor(rank).float())
        self.A = nn.Parameter(th.randn(rank, self.old_linear.weight.shape[1]) * std_dev)
        self.B = nn.Parameter(th.zeros(self.old_linear.weight.shape[0], rank))
        self.alpha: float = alpha
        self.layer: int = layer
        self.head_type: HeadType = head_type

    def forward(self, x_batch: th.Tensor) -> th.Tensor:
        old_pass: th.Tensor = self.old_linear(x_batch)
        new_pass: th.Tensor = self.alpha * ((self.B @ self.A) @ x_batch)
        return old_pass + new_pass

    def get_LoRA_bundle(self) -> LoRABundle:

        return LoRABundle(
            A=self.A,
            B=self.B,
            bias=self.old_linear.bias,
            layer=self.layer,
            alpha=self.alpha,
            head_type=self.head_type
        )

    def get_old_linear(self) -> nn.Linear:
        og_linear_layer: nn.Linear = nn.Linear(
            self.old_linear.weight.shape[1], self.old_linear.weight.shape[0]
        )
        og_linear_layer.weight = self.old_linear.weight
        og_linear_layer.bias = self.old_bias

        return og_linear_layer

