import typing
from enum import Enum

import torch
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
            update_bias: bool = True,
            dropout: float | None = 0.1,
    ) -> None:
        super().__init__()

        self.old_linear: nn.Linear = old_linear
        self.old_bias: nn.Parameter = nn.Parameter(self.old_linear.bias.detach(), requires_grad=False)

        self.old_linear.weight.requires_grad = False
        if update_bias:
            self.old_linear.bias.requires_grad = True
        else:
            self.old_linear.bias.requires_grad = False

        self.alpha: float = alpha
        self.layer: int = layer
        self.head_type: HeadType = head_type
        self.dropout: nn.Module | None = None if dropout is None else nn.Dropout(p=dropout)

        std_dev: float = 1 / torch.sqrt(torch.tensor(rank).float()) # TODO chiedere a biango perche stdev viene calcolata cosi'

        # IMPORTANT: the prefix 'lora_' is there to make lora parameters distinguishable,
        #   so that checkpoints can save just them instead of the whole model.
        #   For this reason, 'lora_' prefix must not be changed/removed
        self.lora_A = nn.Parameter(torch.randn(self.old_linear.weight.shape[1], rank) * std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.old_linear.weight.shape[0]))

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        old_pass: torch.Tensor = self.old_linear(x_batch)

        # Set some of the entries to 0 before feeding them to lora
        if self.dropout is not None:
            x_batch = self.dropout(x_batch)
        new_pass: torch.Tensor = self.alpha * (x_batch @ self.lora_A @ self.lora_B)

        return old_pass + new_pass

    def get_LoRA_bundle(self) -> LoRABundle:
        return LoRABundle(
            A=self.lora_A,
            B=self.lora_B,
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


def lora_state_dict(model: nn.Module, include_biases: bool = True) -> typing.Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if include_biases:
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    else:
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
