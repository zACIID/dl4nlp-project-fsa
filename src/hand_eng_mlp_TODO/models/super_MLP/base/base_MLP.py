from collections import OrderedDict
from typing import Type
from torch import nn, Tensor
from src.hand_eng_mlp_TODO.models.super_MLP.utility.MLP_bundle import MLPBundle
from src.hand_eng_mlp_TODO.models.super_MLP.utility.custom_layers.super_block import SuperBlock


class BaseSuperMLP(nn.Module):
    def __init__(
            self, layers_info: list[MLPBundle], f: Type[nn.Module],
    ) -> None:

        super().__init__()

        self._n_layers: int = len(layers_info)
        self.model: nn.Sequential = self._build_model(layers_info, f)

    def _build_model(self, layers_info: list[MLPBundle], f: Type[nn.Module]) -> nn.Sequential:
        model_layers: OrderedDict = OrderedDict()
        f_counter: int = 1

        for idx, layer_info in enumerate(layers_info):
            layer_type: Type[nn.Module] = layer_info.get_type()
            args: tuple = layer_info.get_bundle()
            model_layers[layer_info.name] = layer_type(*args)

            flag: bool = True
            if self._n_layers - 1 > idx:
                nxt_layer_type: Type[nn.Module] = layers_info[idx + 1].get_type()
                flag = nxt_layer_type != nn.LayerNorm

            if layer_type != nn.Dropout and flag:
                model_layers[f"f_{f_counter}"] = f()
                f_counter += 1

        return nn.Sequential(
            model_layers
        )

    def forward(self, x_batch) -> Tensor:
        return self.model(x_batch)

    def set_superblock_layer(
            self, in_features: int, mid_out: int, out_features: int, idx: int,
            f: Type[nn.Module], layernorm: bool, alpha: float, dropout: float,
            n_layers: int
    ) -> list[MLPBundle]:

        layer_spec: list[MLPBundle] = []

        if (not idx) or (idx == n_layers - 1):
            dropout = .0

        layer_spec.append(MLPBundle(
            index=idx + 1, in_features=in_features, mid_out=mid_out,
            out_features=out_features, alpha=alpha, dropout=dropout,
            l_type=SuperBlock, f=f
        )
        )

        if layernorm:
            layer_spec.append(
                MLPBundle(index=idx + 1, emb_size=out_features, l_type=nn.LayerNorm)
            )

        return layer_spec

    def set_linear_layer(
            self, in_features: int, out_features: int, idx: int,
            layernorm: bool, dropout: float, n_layers: int) -> list[MLPBundle]:

        layer_spec: list[MLPBundle] = []

        if idx >= n_layers * 2:
            dropout = .0

        layer_spec.append(MLPBundle(index=idx + 1, in_features=in_features,
                                    out_features=out_features, l_type=nn.Linear))
        if layernorm:
            layer_spec.append(
                MLPBundle(
                    index=idx + 1, emb_size=out_features,
                    l_type=nn.LayerNorm)
            )

        if dropout > .0:
            layer_spec.append(
                MLPBundle(index=idx + 1, dropout=dropout, l_type=nn.Dropout)
            )

        return layer_spec
