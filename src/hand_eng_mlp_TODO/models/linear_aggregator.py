import torch
import numpy as np
from torch.nn.functional import softmax
from torch import Tensor, randn
from torch.nn import Module, Parameter


class LinAggregator(Module):

    def __init__(
            self, in_features: int, out_features: int,
            MLM_mat: Parameter, MLM_b: Parameter) -> None:
        super().__init__()

        std_dev: float = 1 / out_features

        self._in_features: int = in_features
        self._out_features: int = out_features
        self.MLM_mat: Parameter = MLM_mat if MLM_mat.shape[0] == in_features else MLM_mat.T
        self.MLM_b: Parameter = MLM_b
        self._W: Parameter = Parameter(data=randn(in_features, 1) * std_dev)
        self._b: Parameter = Parameter(data=randn(1) * std_dev)

        self.MLM_mat.requires_grad = False
        self.MLM_b.requires_grad = False

    def forward(self, seq: Tensor) -> Tensor:
        # first dimension is reserved to the batch size
        sm_dim: int = 2  # computing softmax sliding the cols
        m_dim: int = 1  # computing max sliding the rows

        logits: Tensor = softmax(seq @ self.MLM_mat + self.MLM_b, dim=sm_dim)
        seq_weights: Tensor = torch.abs(seq @ self._W + self._b)
        cntx_ten: Tensor = torch.max(logits * seq_weights, dim=m_dim).values

        return self.custom_max_pooling(cntx_ten)

    def custom_max_pooling(self, batch: Tensor):
        evenly_splits: np.ndarray = np.linspace(
            0, self.MLM_mat.shape[1], self._out_features + 1, dtype=np.int32
        )
        chunks: list[int] = [
            evenly_splits[i + 1] - evenly_splits[i]
            for i in range(0, len(evenly_splits) - 1)
        ]

        splits: list[Tensor] = list(
            torch.split(batch,  chunks, dim=-1)
        )

        for idx, chunk in enumerate(splits):
            splits[idx] = torch.max(chunk, dim=-1, keepdim=True).values

        return torch.cat(splits, dim=-1)