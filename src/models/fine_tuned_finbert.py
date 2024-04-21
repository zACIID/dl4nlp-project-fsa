from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertForSequenceClassification,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.tokenization_utils_base import BatchEncoding
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR

import models.modules.lora as lora


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class FineTunedFinBERT(L.LightningModule):
    """
    Class that represents a model (currently "ahmedrachid/FinancialBERT-Sentiment-Analysis" from HuggingFace) combined with
    a custom implemented LoRA fine-tuning infrastructure.
    The caller can choose where to apply LoRA layers (query, key, value, output projection matrices of transformer layers)
    The parameters option should contain the following keys with a valid bool value:
    """

    def __init__(
            self,
            lora_rank: int,
            # TODO Might hardcode this since we are dependent on its implementation (how last layer distributes class labels)
            model_name_or_path: str = "ahmedrachid/FinancialBERT-Sentiment-Analysis",
            one_cycle_max_lr: float = 2e-5,
            weight_decay: float = 0.0,
            lora_alpha: float = 1,
            W_q: bool = True,
            W_v: bool = True,
            W_k: bool = True,
            W_o: bool = True,
            update_bias: bool = True,
            one_cycle_pct_start: float = 0.3,
            **kwargs,
    ):
        """
        :param model_name_or_path: pointing to local model or Huggingface hub
        :param loss_fun: loss function
        :param epochs: number of epochs the model will be trained for - needed by the LR scheduler
        :param n_batches: e.g. obtainable by `len(train_dataloader)` - needed by the LR scheduler
        :param max_lr: maximum learning rate that the LR scheduler will push the optimizer to
        :param weight_decay:
        :param lora_rank: rank of the gradient update matrices used by LoRA. 
            LoRA paper found good values to be 2, 4, 8 on GPT-3
        :param W_q: if True, then all the attention.query layers inside the model's transformer 
            layers will have a LoRa layer
        :param W_v: if True, then all the attention.value layers inside the model's transformer 
            layers will have a LoRa layer
        :param W_k: if True, then all the attention.key layers inside the model's transformer 
            layers will have a LoRa layer
        :param W_o: if True, then all the dense output linear layers inside the model's transformer 
            layers will have a LoRa layer
        :param update_bias: if true, model bias parameters will require gradient, meaning that they will
            be updated during backpropagation. This was mentioned in the LoRA paper to be empirically effective,
            although they (if I recall correctly) said that their study on biases wasn't rigorous
        :param lora_alpha: LoRA hyperparameter that weighs the gradient update matrices
        :param kwargs:
        """

        super().__init__()

        # NOTE: this call saves all the parameters passed to __init__ into self.hparams
        #   For this reason, do not delete the parameters even if they seem unused
        self.save_hyperparameters(logger=True)

        self.model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=3
        )

        self._loRA_layers: dict[lora.HeadType, list[lora.LoRABundle]] = {
            key: [] for key in lora.HeadType
        }
        self._update_bias: bool = update_bias

        self._freeze_net()

        if lora_rank is None or lora_rank < 0:
            raise ValueError('lora_rank must be greater than or equal to 0')

        self._setup_LoRA_layers(
            rank=lora_rank, alpha=lora_alpha, W_q=W_q, W_v=W_v, W_k=W_k, W_o=W_o
        )

    # NOTES ON LOGGING:
    # - PyTorch Lightning already logs useful stuff to the console
    #   In the following ref one can retrieve the lightning console logger and customize it/log directly to it
    #       https://lightning.ai/docs/pytorch/stable/common/console_logs.html
    # - There are multiple hooks such as on_train_start, on_train_epoch_end, etc.
    #   for each phase (train, val, test) that can be leveraged to log various stuff if needed
    # - self.log() has a default reduce_fx=torch.mean argument which makes it so that stuff logged at each
    #   batch is automatically averaged - such param can be overridden if needed
    # - Check autologging ref:
    #   https://lightning.ai/docs/pytorch/stable/extensions/logging.html#logging-from-a-lightningmodule
    #   - PyTorch Lightning auto-detects the logger based on what is passed to the Trainer instance
    #   - can log in multiple places based on the arguments passed to self.log()

    def _freeze_net(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, **inputs) -> MaskedLMOutput:
        return self.model(**inputs)

    def get_LoRA_layers(self) -> dict[lora.HeadType, list[lora.LoRABundle]]:
        """Collection of """
        return self._loRA_layers

    def _setup_LoRA_layers(
            self,
            rank: int,
            alpha: float,
            W_q: bool,
            W_v: bool,
            W_k: bool,
            W_o: bool
    ) -> None:
        for depth, layer in enumerate(self.model.bert.encoder.layer):
            if W_q:
                layer.attention.self.query = self._set_layer(
                    layer=layer.attention.self.query,
                    head_type=lora.HeadType.W_q,
                    depth=depth,
                    rank=rank,
                    alpha=alpha
                )

            if W_v:
                layer.attention.self.value = self._set_layer(
                    layer=layer.attention.self.value,
                    head_type=lora.HeadType.W_v,
                    depth=depth,
                    rank=rank,
                    alpha=alpha
                )

            if W_k:
                layer.attention.self.key = self._set_layer(
                    layer=layer.attention.self.key,
                    head_type=lora.HeadType.W_k,
                    depth=depth,
                    rank=rank,
                    alpha=alpha
                )

            if W_o:
                layer.attention.output.dense = self._set_layer(
                    layer=layer.attention.output.dense,
                    head_type=lora.HeadType.W_o,
                    depth=depth,
                    rank=rank,
                    alpha=alpha
                )

    def _set_layer(
            self,
            layer: nn.Module,
            head_type: lora.HeadType,
            depth: int,
            rank: int,
            alpha: float
    ) -> lora.CustomLoRA:
        if not isinstance(layer, nn.Linear):
            layer = layer.get_old_linear()

        temp_layer: lora.CustomLoRA = lora.CustomLoRA(
            old_linear=layer,
            layer=depth,
            head_type=head_type,
            rank=rank,
            alpha=alpha,
            update_bias=self._update_bias
        )

        self._loRA_layers[head_type].append(temp_layer.get_LoRA_bundle())

        return temp_layer

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        # TODO might want to override this for example to return the sentiment score
        # This function is called during Trainer.predict()
        super().predict_step(args, kwargs)

    def training_step(
            self,
            batch: BatchEncoding,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> torch.Tensor:
        return self._base_step(batch, batch_idx, dataloader_idx, step_type="train")

    def validation_step(
            self,
            batch: BatchEncoding,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> torch.Tensor:
        return self._base_step(batch, batch_idx, dataloader_idx, step_type="val")

    def test_step(
            self,
            batch: BatchEncoding,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> torch.Tensor:
        return self._base_step(batch, batch_idx, dataloader_idx, step_type="test")

    def _base_step(
            self,
            batch: BatchEncoding,
            batch_idx: int,
            dataloader_idx: int = 0,
            step_type: str = None
    ) -> torch.Tensor:
        tokenizer_output, sentiment_score = batch
        outputs: MaskedLMOutput = self(**tokenizer_output)

        # Classes are { 0: bearish, 1: neutral, 2: bullish } for the
        #   ahmedrachid/FinancialBERT-Sentiment-Analysis model
        # Transpose because it is a batch of 3-elements tensors
        probabilities = F.softmax(outputs.logits, dim=0).T
        bearish_prob, bullish_prob = probabilities[0], probabilities[2]

        # This is also how ProsusAI/finbert predicts sentiment score:
        #   positive prob - negative prob, and then it uses MSE loss
        pred_sentiment_score = bullish_prob - bearish_prob
        loss = F.mse_loss(sentiment_score, pred_sentiment_score)

        # TODO define other metrics to log, e.g. taken by torchmetrics or HuggingFace's evaluate package
        self.log_dict(
            dictionary={
                f"{step_type}_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def configure_optimizers(self):
        # TODO "manual optimization is required [meaning that I have to manually call zero_grad, step, etc. on the optimizer | ndr]
        #   when working with multiple optimizers https://lightning.ai/docs/pytorch/stable/common/optimization.html#automatic-optimization
        #   This is important for the idea of class "WithClassificationLayers" that accepts as input one PyTorchLightning module
        #       and will extract the optimizers from it

        # Why no decay for bias and LayerNorm parameters?
        # Possible explanation:
        # https://stats.stackexchange.com/questions/576463/why-not-perform-weight-decay-on-layernorm-embedding
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]

        # SOME IMPORTANT NOTES ON LEARNING RATE:
        # From OneCycleLR docs:
        # The 1cycle learning rate policy changes the learning rate after every batch.
        # step() should be called after a batch has been used for training.
        #
        # Also, according to the outputs int and the answer to this post
        # https://stackoverflow.com/questions/73471929/how-to-use-onecyclelr
        # 1. Starting learning rate provided to the optimizer seems to be ignored
        # 2. max_lr is the maximum learning rate of OneCycleLR.
        #   To be exact, the learning rate will increate from max_lr / div_factor to max_lr
        #   in the first pct_start * total_steps steps,
        #   and decrease smoothly to max_lr / final_div_factor then.

        # REFERENCES: why AdamW + OneCycleLR scheduler?
        # 1. https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
        # 2. https://residentmario.github.io/pytorch-training-performance-guide/lr-sched-and-optim.html
        optimizer = AdamW(optimizer_grouped_parameters, lr=1)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.one_cycle_max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.one_cycle_pct_start
        )

        # Docs on return values and scheduler config dictionary:
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        # As said above, OneCycleLR should be stepped after each optimizer step
        scheduler = {"name": OneCycleLR.__name__, "scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
