from math import ceil
from typing import Type, Callable
from torch import nn
from src.hand_eng_mlp_TODO.models.super_MLP.base.base_MLP import BaseSuperMLP
from src.hand_eng_mlp_TODO.models.super_MLP.utility.MLP_bundle import MLPBundle


class RhomboidMLP(BaseSuperMLP):
    def __init__(
            self, n_layers: int, in_features: int, out_features: int,
            beta: float = 1.5, f: Type[nn.Module] = nn.Tanh, alpha: float = 0.5,
            dropout: float = .2, linear: bool = False, layernorm: bool = False,
            hybrid: bool = False
    ) -> None:

        self.step_comp: Callable[[int, int, int], float] = lambda x, y, z: 2 * (x - y) / (z - 1)

        layers_spec: list[MLPBundle] = []

        mu: int = ceil(in_features * beta)
        amp_aug: float = self.step_comp(mu, in_features, n_layers)

        if linear:
            layers_spec = self._linear_setup(
                n_layers, in_features, out_features, mu, amp_aug, layernorm,
                dropout
            )


        elif hybrid:
            raise NotImplementedError("Bruv I yet have to implement this")

        else:
            layers_spec = self._superblock_setup(
                n_layers, in_features, out_features, mu, amp_aug, f, layernorm,
                alpha, dropout
            )

        super().__init__(layers_spec, f)

    def _superblock_setup(
            self, n_layers: int, in_features: int, out_features: int, mu: int,
            amp_aug: float, f: Type[nn.Module], layernorm: bool, alpha: float,
            dropout: float
    ) -> list[MLPBundle]:

        left_ls: list[MLPBundle] = []
        right_ls: list[MLPBundle] = []

        prev_out: int = in_features
        step_len: int = round(amp_aug)
        idx: int = 0
        right_count: int = n_layers - 1

        flag: bool = False

        while not flag:
            step_len = ceil(amp_aug)
            nxt_out: int = prev_out + step_len

            tup_left: tuple[int, int, int] = (prev_out, nxt_out, nxt_out)
            tup_right: tuple[int, int, int] = (nxt_out, prev_out, prev_out)

            if nxt_out >= mu:
                tup_left = (prev_out, mu, mu)
                tup_right = (mu, prev_out, prev_out)
                flag = True

            left_ls += self.set_superblock_layer(
                *tup_left, idx=idx, f=f, layernorm=layernorm, alpha=alpha,
                dropout=dropout, n_layers=n_layers
            )

            right_ls = self.set_superblock_layer(
                *tup_right, idx=right_count, f=f, layernorm=layernorm,
                alpha=alpha, dropout=dropout, n_layers=n_layers
            ) + right_ls

            prev_out = nxt_out
            idx += 1
            right_count -= 1

        right_ls += self.set_superblock_layer(
            in_features=in_features, mid_out=in_features, out_features=out_features,
            idx=n_layers, f=f, layernorm=False, alpha=alpha, dropout=dropout,
            n_layers=n_layers
        )

        return left_ls + right_ls

    def _linear_setup(
            self, n_layers: int, in_features: int, out_features: int, mu: int,
            amp_aug: float, layernorm: bool, dropout: float) -> list[MLPBundle]:

        left_ls: list[MLPBundle] = []
        right_ls: list[MLPBundle] = []

        prev_out: int = in_features
        step_len: int = round(amp_aug)
        idx: int = 0

        right_count: int = 2 * n_layers - 1

        flag: bool = False

        while not flag:
            nxt_out: int = prev_out + step_len

            tup_left: tuple[int, int] = (prev_out, nxt_out)
            tup_right: tuple[int, int] = (nxt_out, prev_out)

            if nxt_out >= mu:
                tup_left = (prev_out, mu)
                tup_right = (mu, prev_out)
                flag = True

            left_ls += self.set_linear_layer(
                *tup_left, idx=idx, layernorm=layernorm,
                dropout=dropout, n_layers=n_layers
            )

            left_ls += self.set_linear_layer(
                tup_left[1], tup_left[1], idx=idx + 1, layernorm=layernorm,
                dropout=dropout, n_layers=n_layers
            )

            right_ls = self.set_linear_layer(
                tup_right[1], tup_right[1], idx=right_count - 1, layernorm=layernorm,
                dropout=dropout, n_layers=n_layers
            ) + right_ls

            right_ls = self.set_linear_layer(
                *tup_right, idx=right_count, layernorm=layernorm,
                dropout=dropout, n_layers=n_layers
            ) + right_ls

            prev_out = nxt_out
            idx += 2
            right_count -= 2

        right_ls += self.set_linear_layer(
            in_features=in_features, out_features=in_features,
            idx=n_layers * 2, layernorm=False, dropout=dropout,
            n_layers=n_layers
        )

        right_ls += self.set_linear_layer(
            in_features=in_features, out_features=out_features,
            idx=n_layers * 2 + 1, layernorm=False, dropout=dropout,
            n_layers=n_layers
        )

        return left_ls + right_ls
