from typing import Type
from torch import nn
from src.hand_eng_mlp_TODO.models.super_MLP.base.base_MLP import BaseSuperMLP
from src.hand_eng_mlp_TODO.models.super_MLP.utility.MLP_bundle import MLPBundle


class BoxMLP(BaseSuperMLP):
    def __init__(
            self, n_layers: int, in_features: int, out_features: int,
            f: Type[nn.Module] = nn.Tanh, alpha: float = 0.5,
            dropout: float = .2, linear: bool = False, layernorm: bool = False
    ) -> None:

        layers_spec: list[MLPBundle] = []

        if linear:
            layers_spec = self._linear_setup(
                n_layers, in_features, out_features, layernorm, dropout
            )
        else:
            layers_spec = self._superblock_setup(
                n_layers, in_features, out_features, f, layernorm, alpha, dropout
            )

        super().__init__(layers_spec, f)

    def _superblock_setup(
            self, n_layers: int, in_features: int, out_features: int,
            f: Type[nn.Module], layernorm: bool, alpha: float, dropout: float
    ) -> list[MLPBundle]:

        layers_spec: list[MLPBundle] = []

        for idx in range(n_layers - 1):
             layers_spec += self.set_superblock_layer(
                in_features=in_features, mid_out=in_features, out_features=in_features,
                idx=idx, f=f, layernorm=layernorm, alpha=alpha, dropout=dropout,
                n_layers=n_layers
            )

        layers_spec += self.set_superblock_layer(
                in_features=in_features, mid_out=in_features, out_features=out_features,
                idx=n_layers - 1, f=f, layernorm=False, alpha=alpha, dropout=dropout,
                n_layers=n_layers
            )

        return layers_spec

    def _linear_setup(
            self, n_layers: int, in_features: int, out_features: int,
            layernorm: bool, dropout: float) -> list[MLPBundle]:

        layers_spec: list[MLPBundle] = []

        for idx in range(n_layers * 2 - 1):

            layers_spec += self.set_linear_layer(
                 in_features=in_features, out_features=in_features,
                idx=idx, layernorm=layernorm, dropout=dropout, n_layers=n_layers
            )

        layers_spec += self.set_linear_layer(
                in_features=in_features, out_features=in_features,
                idx=n_layers * 2 - 1, layernorm=layernorm, dropout=dropout,
                n_layers=n_layers
        )

        layers_spec += self.set_linear_layer(
                in_features=in_features, out_features=out_features,
                idx=n_layers * 2, layernorm=False, dropout=dropout,
                n_layers=n_layers
        )

        return layers_spec