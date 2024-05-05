import typing
from typing import Any, Mapping

import lightning as L
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import models.modules.lora as lora
import models.loss_functions as loss_functions


# fucking shit models
# PRE_TRAINED_MODEL_PATH = "ahmedrachid/FinancialBERT-Sentiment-Analysis" # TODO trying to change
# PRE_TRAINED_MODEL_PATH = "ipuneetrathore/bert-base-cased-finetuned-finBERT"

PRE_TRAINED_MODEL_PATH = "ProsusAI/finbert"


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class FineTunedFinBERT(L.LightningModule):
    """
    Class that represents a pre-trained sequence classification model `from combined
        with a custom implemented LoRA fine-tuning infrastructure.
    The caller can choose where to apply LoRA layers
        (query, key, value, output projection matrices of transformer layers)
    """

    def __init__(
            self,
            lora_rank: int = 64,
            one_cycle_max_lr: float = 2e-5,
            weight_decay: float = 0.0,
            lora_alpha: float = 1,
            lora_dropout: float = 0.1,
            W_q: bool = True,
            W_v: bool = True,
            W_k: bool = True,
            W_o: bool = True,
            W_pooler: bool = True,
            W_classifier: bool = True,
            update_bias: bool = True,
            C: float = 1.0,
            one_cycle_pct_start: float = 0.3,
            log_hparams: bool = True,
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
        self.save_hyperparameters()

        # NOTE: need to manually call log_params here because mlflow.pytorch.autolog() doesn't log them
        #   and I am not using MLflowLogger because it apparently duplicates logging when autolog() is active
        # NOTE2: synchronous false because this hangs if log_params has already been called for the current run
        #   I would like it to fail silently instead of blocking my training until it reaches the timeout
        if log_hparams:
            mlflow.log_params(self.hparams, synchronous=False)

        # self.model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
        #     PRE_TRAINED_MODEL_PATH,
        #     num_labels=3
        # ) # TODO trying other model
        self.model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_PATH)
        self._update_bias: bool = update_bias
        self._val_predictions: typing.List[torch.Tensor] = []
        self._val_targets: typing.List[torch.Tensor] = []

        self._freeze_net()

        if lora_rank is None or lora_rank < 0:
            raise ValueError('lora_rank must be greater than or equal to 0')

        self._setup_lora_layers(
            rank=lora_rank,
            alpha=lora_alpha,
            W_q=W_q,
            W_v=W_v,
            W_k=W_k,
            W_o=W_o,
            W_pooler=W_pooler,
            W_classifier=W_classifier
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

    def forward(self, **inputs) -> SequenceClassifierOutput:
        return self.model(**inputs)

    def _to_sentiment_score(self, output: SequenceClassifierOutput) -> torch.Tensor:
        # NOTE:
        # Classes are { 0: bearish, 1: neutral, 2: bullish } for the
        #   ahmedrachid/FinancialBERT-Sentiment-Analysis model
        # Classes are { 0: positive, 1: negative, 2: neutral } for the
        #   ProsusAI/finbert model

        # Transpose because it is a batch of 3-elements tensors
        probabilities = F.softmax(output.logits, dim=1).T
        # bearish_prob, bullish_prob = probabilities[0], probabilities[2] # TODO for ahmedrachid
        bearish_prob, bullish_prob = probabilities[1], probabilities[0]

        # This is also how ProsusAI/finbert predicts sentiment score:
        #   positive prob - negative prob, and then it uses MSE loss
        pred_sentiment_score = bullish_prob - bearish_prob
        return pred_sentiment_score

    def _setup_lora_layers(
            self,
            rank: int,
            alpha: float,
            W_q: bool,
            W_v: bool,
            W_k: bool,
            W_o: bool,
            W_pooler: bool,
            W_classifier: bool
    ) -> None:
        for layer in self.model.bert.encoder.layer:
            if W_q:
                layer.attention.self.query = self._set_layer(
                    layer=layer.attention.self.query,
                    rank=rank,
                    alpha=alpha
                )

            if W_v:
                layer.attention.self.value = self._set_layer(
                    layer=layer.attention.self.value,
                    rank=rank,
                    alpha=alpha
                )

            if W_k:
                layer.attention.self.key = self._set_layer(
                    layer=layer.attention.self.key,
                    rank=rank,
                    alpha=alpha
                )

            if W_o:
                layer.attention.output.dense = self._set_layer(
                    layer=layer.attention.output.dense,
                    rank=rank,
                    alpha=alpha
                )

        if W_pooler:
            self.model.bert.pooler.dense = self._set_layer(
                layer=self.model.bert.pooler.dense,
                rank=rank,
                alpha=alpha
            )

        if W_classifier:
            self.model.bert.classifier = self._set_layer(
                layer=self.model.classifier,
                rank=rank,
                alpha=alpha
            )

    def _set_layer(
            self,
            layer: nn.Linear,
            rank: int,
            alpha: float
    ) -> lora.CustomLoRA:
        temp_layer: lora.CustomLoRA = lora.CustomLoRA(
            old_linear=layer,
            rank=rank,
            alpha=alpha,
            dropout=self.hparams.lora_dropout,
            update_bias=self._update_bias
        )

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

        outputs = self.forward(**tokenizer_output)
        pred_sentiment_score = self._to_sentiment_score(outputs)

        mse = F.mse_loss(sentiment_score, pred_sentiment_score)
        mae = F.l1_loss(sentiment_score, pred_sentiment_score)
        is_sign_correct = torch.sum(
            loss_functions.sign_accuracy_mask(sentiment_score, pred_sentiment_score)
        ) / len(sentiment_score)
        loss = loss_functions.custom_regression_loss(sentiment_score, pred_sentiment_score, C=self.hparams.C)

        if step_type == 'val':
            self._val_predictions.append(pred_sentiment_score)
            self._val_targets.append(batch[1])

        self.log_dict(
            dictionary={
                f"{step_type}_loss": loss,
                f"{step_type}_mse": mse,
                f"{step_type}_mae": mae,
                f"{step_type}_sign_accuracy": is_sign_correct,
                f"{step_type}_positive_predictions": loss_functions.sign_accuracy_mask(
                    torch.ones(sentiment_score.shape, device=self.device), pred_sentiment_score
                ).sum() / len(sentiment_score),
                f"{step_type}_mean_prediction": pred_sentiment_score.mean(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        predictions = torch.concatenate(self._val_predictions, dim=0).flatten()
        targets = torch.concatenate(self._val_targets, dim=0).flatten()

        # NOTE: SemEval2017 Task 5 uses weighted cosine similarity to account for
        #   teams that chose not to predict the whole test set.
        #   The weight is 1 if an attempt to predict the whole test set is made, so,
        #       in our case, weight calculation can be omitted
        cosine_similarity = F.cosine_similarity(predictions, targets, dim=0)
        self.log(
            name=f"val_cosine_similarity",
            value=cosine_similarity,
            on_step=False,  # True raises error because we are on epoch end
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

    def configure_optimizers(self):
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, weight_decay=self.hparams.weight_decay)

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

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # NOTE: by overriding this, lightning's Trainer automatic checkpointing stores only the lora stuff,
        #   meaning that checkpoint size is greatly reduced
        # To use these checkpoints, the model has to first be normally instantiated
        return lora.lora_state_dict(self.model, include_biases=self._update_bias)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # Needed to apply the state_dict to the actual model
        # NOTE: if the state dict is the lora_state_dict, a warning about "missing keys" will be raised by lightning:
        #   we can safely ignore it, because we are on purpose not saving the original parameters
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)
