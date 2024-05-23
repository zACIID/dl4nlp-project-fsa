from typing import Type
from torch import nn
from src.hand_eng_mlp_TODO.models.super_MLP.base.base_MLP import BaseSuperMLP
from src.hand_eng_mlp_TODO.models.super_MLP.utility.MLP_bundle import MLPBundle


class RepRomboidMLP(BaseSuperMLP):
    def __init__(self, n_layers: int, in_features: int, out_features: int,
                 beta: float = 2., f: Type[nn.Module] = nn.Tanh, alpha: float = 0.5,
                 dropout: float = .2, linear: bool = False, layernorm: bool = False) -> None:

        layers_spec: list[MLPBundle] = []

        mid_out: int = int(beta * in_features)

        if linear:
            layers_spec = self._linear_setup(
                n_layers, in_features, out_features, mid_out, layernorm, dropout
            )

        else:
            layers_spec = self._superblock_setup(
                n_layers, in_features, out_features, mid_out,
                f, layernorm, alpha, dropout
            )

        super().__init__(layers_spec, f)

    def _linear_setup(
            self, n_layers: int, in_features: int, out_features: int, mid_out: int,
            layernorm: bool, dropout: float) -> list[MLPBundle]:

        layers_spec: list[MLPBundle] = []

        for idx in range(n_layers * 2 - 1):

            if not idx % 2:
                layers_spec += self.set_linear_layer(
                    in_features=in_features, out_features=mid_out,
                    idx=idx, layernorm=layernorm, dropout=dropout, n_layers=n_layers
                )
            else:
                layers_spec += self.set_linear_layer(
                    in_features=mid_out, out_features=in_features,
                    idx=idx, layernorm=layernorm, dropout=dropout, n_layers=n_layers
                )

        layers_spec += self.set_linear_layer(
            in_features=mid_out, out_features=in_features,
            idx=n_layers * 2 - 1, layernorm=layernorm, dropout=dropout,
            n_layers=n_layers
        )

        layers_spec += self.set_linear_layer(
            in_features=in_features, out_features=out_features,
            idx=n_layers * 2, layernorm=False, dropout=dropout,
            n_layers=n_layers
        )

        return layers_spec

    def _superblock_setup(
            self, n_layers: int, in_features: int, out_features: int, mid_out: int,
            f: Type[nn.Module], layernorm: bool, alpha: float, dropout: float
    ) -> list[MLPBundle]:

        layers_spec: list[MLPBundle] = []

        for idx in range(n_layers - 1):
            layers_spec += self.set_superblock_layer(
                in_features=in_features, mid_out=mid_out,
                out_features=in_features, idx=idx, f=f, layernorm=layernorm,
                alpha=alpha, dropout=dropout, n_layers=n_layers
            )

        layers_spec += self.set_superblock_layer(
            in_features=in_features, mid_out=mid_out, out_features=out_features,
            idx=n_layers - 1, f=f, layernorm=False, alpha=alpha, dropout=dropout,
            n_layers=n_layers
        )

        return layers_spec
