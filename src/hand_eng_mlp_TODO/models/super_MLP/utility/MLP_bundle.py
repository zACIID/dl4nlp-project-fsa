from typing import Type
import torch.nn as nn


class MLPBundle:
  def __init__(
      self, l_type: Type[nn.Module], index: int, in_features: int = None,
      out_features: int = None, mid_out: int = None, alpha: float = None,
      f: Type[nn.Module] = None, dropout: float = .0, emb_size: int = None
  ) -> None:

    self._index: int = index

    match l_type:
      case nn.Dropout:
        self._type: Type[nn.Dropout] = nn.Dropout
        self.name: str = f"dropout_{self._index}"
        self._bundle: tuple[float] = (dropout, )

      case nn.LayerNorm:
        self._type: Type[nn.LayerNorm] = nn.LayerNorm
        self.name = f"layernorm_{self._index}"
        self._bundle: tuple[float] = (emb_size, )

      case nn.Linear:
        self._type: Type[nn.Linear] = nn.Linear
        self.name = f"linear_{self._index}"
        self._bundle: tuple[int, int] = (in_features, out_features)

      case SuperBlock:
        self._type: Type[SuperBlock] = SuperBlock
        self.name: str = f"superblock_{self._index}"
        self._bundle: tuple = (
            in_features, mid_out, out_features, alpha,
            f, dropout
        )

  def get_bundle(self) -> tuple:
    return self._bundle

  def get_type(self) -> Type[nn.Module]:
    return self._type
