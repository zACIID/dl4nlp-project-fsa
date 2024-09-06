import logging
import typing
import warnings
from typing import Any, Mapping

import lightning as L
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification
)
from mlflow.entities.model_registry import ModelVersion
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import training.loader as loader
import utils.mlflow_env as env
import fine_tuned_finbert.models.fine_tuned_finbert as ft
from hand_eng_mlp_TODO.models.model_beijin import ModelBeijin

# TODO move this "loss_function" module to utils/ maybe
from fine_tuned_finbert.models.loss_functions import sign_accuracy_mask


class EndToEndModel(L.LightningModule):
    """
    # TODO maybe we should call this EnsembleModel and that's it
    """

    def __init__(
            self,
            finbert: L.LightningModule,
            hemlp: L.LightningModule,
            log_hparams: bool = True,
            # TODO n_layers; in_features are determined by the two models size, out_features may be a hyperparam
            n_layers: int,
            **kwargs,
    ):
        """
        :param finbert_init_args:
        :param hemlp_init_args:
        :param log_hparams:
        :param kwargs:
        """

        super().__init__()

        # NOTE: this call saves all the parameters passed to __init__ into self.hparams
        #   For this reason, do not delete the parameters even if they seem unused
        self.save_hyperparameters()


        # TODO finbert and hemlp are passed from the outside, i.e. inside loader.py
        # TODO we finally opted for common linear layers so that tuning and impl. is easier

        self.finbert: ft.FineTunedFinBERT | None = None
        self.hemlp: hemlp.ModelBeijin | None = None
        self.aggregation_layers = torch.Sequential(
            # TODO need to implement here
            # TODO maybe I can even use a boxmlp...,
            # TODO add Tanh() layer at the end to get one score \in [-1, 1], maybe even a Flatten() before it is needed
        )

        # NOTE: need to manually call log_params here because mlflow.pytorch.autolog() doesn't log them
        #   and I am not using MLFlowLogger because it apparently duplicates logging when autolog() is active
        # NOTE2: synchronous false because this hangs if log_params has already been called for the current run
        #   I would like it to fail silently instead of blocking my training until it reaches the timeout
        if log_hparams:
            mlflow.log_params(self.hparams, synchronous=False)

    def setup(self, stage: str) -> None:
        self.finbert = self._load_best_model(loader.Model.FINBERT)
        self.hemlp = self._load_best_model(loader.Model.HAND_ENG_MLP)

        # TODO remove last layer (the classification head) from finbert
        # TODO remove the last activation layer from the MLP
        # TODO freeze the two models: we want to train only the aggregation mlp
        raise NotImplementedError()

    def _load_best_model(self, model: loader.Model):
        pytorch_logger = logging.getLogger("lightning.pytorch")
        pytorch_logger.setLevel(logging.INFO)

        model_name = env.get_registered_model_name(model)
        # alias = env.BEST_TUNED_MODEL_ALIAS
        alias = env.BEST_FULL_TRAINED_MODEL_ALIAS
        client = mlflow.tracking.MlflowClient()
        best_version: ModelVersion = client.get_model_version_by_alias(name=model_name, alias=alias)

        mlflow.set_tag(key='model_name', value=model_name)
        mlflow.set_tag(key='model_alias', value=alias)
        mlflow.set_tag(key='model_version', value=best_version.version)

        best_model: L.LightningModule = mlflow.pytorch.load_checkpoint(
            ft.FineTunedFinBERT if model == loader.Model.FINBERT else hemlp.ModelBeijin,
            best_version.run_id,
            kwargs={
                'strict': False,  # Needed because in FinBert, LoRA checkpoint do not include all model parameters
                'log_hparams': True
            }
        )
        return best_model

    def forward(self, batch) -> SequenceClassifierOutput:
        # TODO implement
        tokenizer_output, embeddings, new_features = batch
        finbert_out = self.finbert(**tokenizer_output)
        hemlp_out = self.finbert(**tokenizer_output)

        # TODO correct?
        return self.aggregation_layers(torch.concatenate([finbert_out, hemlp_out]))

    def predict(self, batch) -> torch.Tensor:
        self.eval()  # Call this explicitly because this is external to PytorchLightning
        with torch.no_grad():
            output = self.forward(batch)
            return self._to_sentiment_score(output)

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) > 0:
            warnings.warn(f"Args are ignored by {self.__class__.__name__}, "
                          f"make sure to pass a pre-trained-BERT-compatible dictionary")

        return self.predict(**kwargs)

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
        tokenizer_output, embeddings, new_features, scores = batch

        pred_scores = self.forward((tokenizer_output, embeddings, new_features))

        mse = F.mse_loss(scores, pred_scores)
        mae = F.l1_loss(scores, pred_scores)
        sign_accuracy = torch.sum(
            sign_accuracy_mask(scores, pred_scores)
        ) / len(scores)
        loss: torch.Tensor = -torch.cosine_similarity(scores, pred_scores, dim=-1)

        if step_type == 'val':
            self._val_predictions.append(pred_scores)
            self._val_targets.append(batch[1])

        self.log_dict(
            dictionary={
                f"{step_type}_loss": loss,
                f"{step_type}_mse": mse,
                f"{step_type}_mae": mae,
                f"{step_type}_sign_accuracy": sign_accuracy,
                f"{step_type}_positive_predictions": sign_accuracy_mask(
                    torch.ones(scores.shape, device=self.device), pred_scores
                ).sum() / len(scores),
                f"{step_type}_mean_prediction": pred_scores.mean(),
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
            # I want the scheduler to act on just the first fraction of the learning epochs,
            #   so that the last ones are done with a low LR
            # TODO cannot do this as step is still called after the limit is reached - solution I can think of is to manually call step,
            #   but it's a pain in the ass to manually implement the use of optimziers I think
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
        # TODO here only the final aggregation mlp needs to be checkpointed
        raise NotImplementedError()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # Needed to apply the state_dict to the actual model
        # TODO load the mlp state dict
        raise NotImplementedError()
