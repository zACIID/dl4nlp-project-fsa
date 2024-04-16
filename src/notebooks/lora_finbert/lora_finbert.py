# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="JKPGam770FXb"
# ## FinancialBERT

# %% [markdown] id="t62U8Km5DamU"
# *References*
# 1. [FinanacialBERT Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForMaskedLM)
# 2. [Tokenizer Documentation](https://huggingface.co/docs/transformers/en/internal/tokenization_utils)

# %% id="YScJEQBL0Dx0"
import torch as th
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerBase, BertForMaskedLM
import transformers.tokenization_utils_base as ttu
import transformers.modeling_outputs as tm
from transformers.models.bert.modeling_bert import BertLayer

# %% id="M5S50x95BrWg"
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("ahmedrachid/FinancialBERT")
model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained("ahmedrachid/FinancialBERT")

# %% [markdown] id="4qbqr5w5D2tU"
# ## Tokenize sequences

# %% [markdown] id="yvJ8SjlIKNLZ"
# So here given a string or a batch of string (list[str]) the tokenizer return an object containing:
# 1. `input_ids` (plus two for [CLS] and [SEP] if not present)
# 2. `token_type_ids` needed to understand if it's a normal word or a special token
# 3. `attention_mask` needed to understand if the model should accept at certain position the token [MASK]

# %% id="9oUSNeziCQxY" colab={"base_uri": "https://localhost:8080/"} outputId="e048e59d-235c-4ec7-b83c-8d37235ff8a8"
fst_sentence: str = "let's touch them kids"

fst_sample: ttu.BatchEncoding = tokenizer(text=fst_sentence, return_tensors="pt") # bruv pt stands for pytorch

fst_sample

# %% id="QO-WGK9VLeFx" colab={"base_uri": "https://localhost:8080/"} outputId="02e1e390-e396-4e4c-f531-65acce85829d"
sdn_sentence: str = "hope the loss goes down"

# Had to add padding parameter to adjust the lenght of all the sentences to the longest one
sdn_sample: ttu.BatchEncoding = tokenizer(text=[fst_sentence, sdn_sentence], padding=True, return_tensors="pt")

sdn_sample

# %% [markdown] id="YImdtExGECub"
# ## Feeding the model

# %% [markdown] id="dIffsmhhPpFO"
# The model return a matrix of size $(d\times t \times |V|)$ where $d$ is the batch size $t$ is the number of token and $|V|$ is the cardinality of the vocabulary

# %% id="0ONKJserED12" colab={"base_uri": "https://localhost:8080/"} outputId="aa686086-1336-4227-f59d-d50ab5ee1aee"
with th.no_grad():
    logits: th.Tensor = model(**fst_sample).logits  # with this I can have the raw logits before softmax

logits.shape

# %% id="b1CkN23l5QNZ" colab={"base_uri": "https://localhost:8080/"} outputId="695cde8c-954d-4444-d3c9-79d77bd8ca42"
with th.no_grad():
    logits: th.Tensor = model(**sdn_sample).logits  # with this i can have the raw logits before softmax

logits.shape

# %% [markdown] id="7vdJz8dc7NZS"
# Clearly we can shove all those logits in our butt, what we do care are the embeddings $e_i \in \mathbb{R}^{H}$ $\forall i=1, \dots, T$. In order to get them we just need to set the parameter `output_hidden_states=True`, and the model will add inside the object of type `tm.MaskedLMOutput` the `hidden_states` field. As written in the documentation the `hidden_states` is of type `tuple(torch.FloatTensor)`, where the size of tuple is 13 or 25 (because of the 12 or 24 encoders in the architecture each of which output its own embeddings plus the initial embeddings). Each entry of the tuple is of size $(d \times T \times H)$ where $H$ is the embedding size (depending on which model FinBERT was built on, in case of $BERT_{LARGE}$ $H=1024$ otherwise with $BERT_{SMALL}$ $H=768$)

# %% id="04Sj-qBPQUHE" colab={"base_uri": "https://localhost:8080/"} outputId="d1baae7a-2bf5-451f-b316-fb7c5c1fa1af"
with th.no_grad():
    fst_res: tm.MaskedLMOutput = model(**fst_sample, output_hidden_states=True)

fst_res.hidden_states[12].shape

# %% id="L1rLhpnK-ZlL" colab={"base_uri": "https://localhost:8080/"} outputId="4c89f44a-9a4c-4a79-eafa-4e5649d412d1"
with th.no_grad():
    sdn_res: tm.MaskedLMOutput = model(**sdn_sample, output_hidden_states=True)

sdn_res.hidden_states[12].shape

# %% [markdown] id="kQ63IrhdjJpV"
# Let's take a look at the structure of the model:
#
#

# %% id="Rcwz2cmsjQh4" colab={"base_uri": "https://localhost:8080/"} outputId="62c42430-b1a6-4cbd-c328-0a8338f31c84"
model

# %% [markdown] id="Zvp_ePuwsbWA"
# Ey bro chill don't worry about the last module `BertOnlyMLMHead` that thing is only used during pretraining for the masked token

# %% [markdown] id="Ig1280RTZELZ"
# ## LoRa

# %% [markdown] id="QgxpXfpHT2TP"
# ![](https://storage.googleapis.com/lightning-avatars/litpages/01hmchy7r24jjmg7ez52nrm1h1/Screenshot%202024-01-17%20at%203.31.43%E2%80%AFPM.png)

# %% [markdown] id="Iy-u5YsQZMIC"
# *References*
# 1. [LoRA paper](https://arxiv.org/pdf/2106.09685.pdf)
# 2. [LoRa from scratch (lightning off.)](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch)
# 3. [Fine-tuning BERT with simple lightning](https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html)

# %% [markdown] id="-aDjYncclcA6"
# I will implement the fine tuned mode using homemade LoRa, I need to customize two parameters:
# 1. r: rank of each matrix (usually from 2-8)
# 2. The positions where I'm going to add LoRa, allegedly the paper showed the best result were found in applying LoRa just to $W_q$ (projection matrix for queries in the multi head attention) and $W_v$ (projection matrix for values in the multi head attention).
# 3. if the bias should be updatable or not
#

# %% [markdown] id="lEdG8UJPjnf9"
# #### LoRA Bundle
#
# This class is used to represent a LoRA layer, it will contain the minimal parameters to reconstruct a `CustomLoRA` layer

# %% id="9HJ1TWXKUPsW"
from enum import Enum

# Define an enum class
class HeadType(Enum):
    W_q = 1
    W_k = 2
    W_v = 3
    W_o = 4

class LoRABundle:
  def __init__(self, A: nn.Parameter, B: nn.Parameter, bias: nn.Parameter,
                layer: int, alpha: int, head_type: HeadType) -> None:
      self.A = A
      self.B = B
      self.bias = bias
      self.layer = layer
      self.alpha = alpha
      self.head_type = head_type

  def __iter__(self):
    yield self.A
    yield self.B
    yield self.bias
    yield self.layer
    yield self.alpha
    yield self.head_type


# %% [markdown] id="fRTsVem5j_Kq"
# #### CustomLoRA
#
#  Class wrapping the functionalities a LoRa linear layer should have, the `old_linear` layer parameter is mandatory. The caller can either choose to pass a `LoRABundle` object, or free parameters. When calling the constructor with the bundle the class assume the caller had already instantiated a `CustomLoRA` layer in the past. Beware this LoRA implementation by default does NOT freeze the bias of the `old_linear` layer, in the paper they left the bias freezed as the `old_linear.weight`.

# %% id="WJ3aI4yMgAv8"
class CustomLoRA(nn.Module):

  def __init__(self, old_linear: nn.Linear, layer: int = None,
               head_type: HeadType = None,  rank: int = 1, alpha: int = 1,
               bundle: LoRABundle = None, update_bias: bool = True) -> None:

    super().__init__()

    self.old_linear: nn.Linear = old_linear
    self.old_bias: nn.Parameter = nn.Parameter(self.old_linear.bias.detach(), requires_grad=False)

    self.old_linear.weight.requires_grad = False # should have aleardy been done by TunedBERT
    if update_bias:
      self.old_linear.bias.requires_grad = True
    else:
      self.old_linear.bias.requires_grad = False

    print(self.old_linear.bias.requires_grad)

    if bundle is not None:
      self.A: nn.Parameter = bundle.A
      self.B: nn.Parameter = bundle.B
      self.head_type: HeadType = bundle.head_type
      self.alpha: int = bundle.alpha
      self.layer: int = bundle.layer
      self.old_linear.bias = bundle.bias

    else:
      std_dev: float = 1 / th.sqrt(th.tensor(rank).float())
      self.A = nn.Parameter(th.randn(rank, self.old_linear.weight.shape[1]) * std_dev)
      self.B = nn.Parameter(th.zeros(self.old_linear.weight.shape[0], rank))
      self.alpha: int = alpha
      self.layer: int = layer
      self.head_type: HeadType = head_type


  def forward(self, x_batch: th.Tensor) -> th.Tensor:
    old_pass: th.Tensor = self.old_linear(x_batch)
    new_pass: th.Tensor = self.alpha * ((self.B @ self.A) @ x_batch)
    return old_pass + new_pass


  def get_LoRA_bundle(self) -> LoRABundle:

    return LoRABundle(
        A=self.A, B=self.B, bias=self.old_linear.bias, layer=self.layer,
        alpha=self.alpha, head_type=self.head_type
    )


  def get_old_linear(self) -> nn.Linear:

    og_linear_layer: nn.Linear = nn.Linear(
        self.old_linear.weight.shape[1], self.old_linear.weight.shape[0]
    )
    og_linear_layer.weight = self.old_linear.weight
    og_linear_layer.bias = self.old_bias

    return og_linear_layer

# %% [markdown] id="Atm01Q-vlGJc"
# #### TunedBERT
#
# This class represent `financialBERT` model with `CustomLoRA` layers on top, the caller can choose which set up is needed to set up `LoRA` layers. The parameters option should contain the following key with a valid bool value:
#
# - if `W_o` is true then all the dense output linear layers inside the model will have a LoRa layer (all transformers)
# - If `W_k` is true than we will just set LoRa layer for all the `W_k`'s (all transformers)
# - If `W_q` is true than we will just set LoRa layer for all the `W_q`'s (all transformers)
# - If `W_v` is true than we will just set LoRa layer for all the `W_v`'s (all transformers)
# - `rank` is an hyper-parameter defining the dimensionality reduction
# - `alpha` is an hyper-parameter used as scaling factor applied to the matrix.
#
# The class should return a dictionary (`dict[HeadType, LoRAbundle]`) describing the LoRA layers inside the input model. From a `dict[HeadType, LoRAbundle]` the class should be able to reconstruct a set up model starting from the dictionary received as input.

# %% id="fchWgvrM7rl1"
import torch.nn as nn
from functools import partial
from collections.abc import Iterable
from typing import Union, Optional, Callable
from collections import OrderedDict


class TunedBERT(nn.Module):

  @staticmethod
  def build_from_bundles(BERT: nn.Module, loRA_layers: dict[HeadType, list[LoRABundle]]) -> "TunedBERT":
    tuned_BERT: nn.Module = TunedBERT(BERT=BERT, rank=0)
    tuned_BERT.update_LoRA_layers(loRA_layers, remove_old=False)

    return tuned_BERT


  def __init__(
      self, BERT: nn.Module, rank: int, W_q: bool=True, W_v: bool=True,
      W_k: bool=True, W_o: bool=True, upd_bias: bool=True, alpha: float = 1) -> None:

    super().__init__()
    self.__tuned_BERT: nn.Module = BERT
    self.__loRA_layers: dict[HeadType, list[LoRABundle]] = {
        key: [] for key in HeadType
    }
    self.__update_bias: bool = upd_bias

    self.__freeze_net()

    if rank > 0:
      self.__setup_LoRA_layers(
          rank=rank, alpha=alpha, W_q=W_q, W_v=W_v, W_k=W_k, W_o=W_o
      )


  def update_LoRA_layers(
    self, loRA_layers: dict[HeadType, list[LoRABundle]], remove_old: bool = True
  ) -> None:

    if remove_old:
      self.__remove_LoRA_layers()

    self.__apply_action(loRA_layers, to_delete=False, to_update=True)


  def get_LoRA_layers(self) -> dict[HeadType, list[LoRABundle]]:
    return self.__loRA_layers


  def forward(self, x_batch: th.Tensor) -> th.Tensor:
    return self.__tuned_BERT(x_batch)


  def __freeze_net(self) -> None:
    for param in self.__tuned_BERT.parameters():
      param.requires_grad = False


  def __setup_LoRA_layers(
      self, rank: int, alpha: float,
      W_q: bool, W_v: bool, W_k:bool, W_o: bool) -> None:

    for depth, layer in enumerate(self.__tuned_BERT.bert.encoder.layer):

      if W_q:
        layer.attention.self.query = self.__set_layer(
            layer=layer.attention.self.query, head_type=HeadType.W_q,
            depth=depth, rank=rank, alpha=alpha
        )

      if W_v:
        layer.attention.self.value = self.__set_layer(
            layer=layer.attention.self.value, head_type=HeadType.W_v,
            depth=depth, rank=rank, alpha=alpha
        )

      if W_k:
        layer.attention.self.key = self.__set_layer(
            layer=layer.attention.self.key, head_type=HeadType.W_k,
            depth=depth, rank=rank, alpha=alpha
        )

      if W_o:
        layer.attention.output.dense = self.__set_layer(
            layer=layer.attention.output.dense, head_type=HeadType.W_o,
            depth=depth, rank=rank, alpha=alpha
        )


  def __set_layer(self, layer: nn.Module, head_type: HeadType, depth: int,
                  rank: int, alpha: int) -> CustomLoRA:

    if not isinstance(layer, nn.Linear):
      layer = layer.get_old_linear()

    temp_layer: CustomLoRA = CustomLoRA(
            old_linear=layer, layer=depth, head_type=head_type, rank=rank,
            alpha=alpha, update_bias=self.__update_bias
        )

    self.__loRA_layers[head_type].append(temp_layer.get_LoRA_bundle())

    return temp_layer


  def __apply_action(self, loRA_layers: dict[HeadType, list[LoRABundle]],
                     to_delete: bool = True, to_update: bool = True) -> None:

    for depth, layer in enumerate(self.__tuned_BERT.bert.encoder.layer):

      layer.attention.self.query = self.__apply(
        loRA_layers, HeadType.W_q, depth, layer.attention.self.query,
        to_delete, to_update
      )

      layer.attention.self.value = self.__apply(
          loRA_layers, HeadType.W_v, depth, layer.attention.self.value,
          to_delete, to_update
        )

      layer.attention.self.key = self.__apply(
          loRA_layers, HeadType.W_k, depth, layer.attention.self.key,
          to_delete, to_update
        )

      layer.attention.output.dense = self.__apply(
          loRA_layers, HeadType.W_o, depth, layer.attention.output.dense,
          to_delete, to_update
        )

    if to_delete:
      loRA_layers = {
          key: [] for key in HeadType
      }


  def __apply(self, loRA_layers: dict[HeadType, list[LoRABundle]],
              head_type: HeadType, depth: int, layer: nn.Module,
              to_delete: bool = True, to_update: bool = False) -> nn.Module:

    temp_layer: nn.Module = layer

    if (loRA_layers[head_type] and
        len(loRA_layers[head_type]) > depth):

          if to_update:

            if not isinstance(layer, nn.Linear):
              old_linear: nn.Linear = layer.get_old_linear()
            else:
              old_linear: nn.Linear = layer

            temp_layer = CustomLoRA(
                old_linear=old_linear, bundle=loRA_layers[head_type][depth]
            )

            if len(self.__loRA_layers[head_type]) > depth:
              self.__loRA_layers[head_type][depth] = loRA_layers[head_type][depth]

            else:
              self.__loRA_layers[head_type].append(loRA_layers[head_type][depth])

          else:
            temp_layer = layer.get_old_linear()

    return temp_layer


  def __remove_LoRA_layers(self) -> None:
    self.__apply_action(self.__loRA_layers, to_delete=True, to_update=False)

# %% [markdown] id="cauks6zGmpyW"
# ## Full left model
#
# This class will be used to represent the left part of our ensemble approach, the model itself will be composed by:
# - `TunedBERT` object
# - `nn.Linear(in_features=H, out_features=1)` where $H$ is the size of the token `<CLS>` embedding (in our case $768$)
#
# The loss function and activation function will be received as input so I don't have to bear the weight of a choice.

# %% id="zBZv4O_gLQIS"
# TODO MORE
