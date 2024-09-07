import collections
import warnings
from typing import Any, Mapping

import lightning as L
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers.tokenization_utils_base import BatchEncoding

import fine_tuned_finbert.models.fine_tuned_finbert as ft
from fine_tuned_finbert.models.loss_functions import sign_accuracy_mask
from fine_tuned_finbert.models.modules.lora import CustomLoRA
# TODO move this "loss_function" module to utils/ maybe
from hand_eng_mlp_TODO.models.model_beijin import ModelBeijin


class EndToEndModel(L.LightningModule):
    """
    # TODO maybe we should call this EnsembleModel and that's it
    """

    _OUT_FEATURES = 768
    """
    Fixed number of out_features for linear layers;
    in_features is defined by the sum of the out_features of the last layers
      of the finbert and hemlp models
    """

    def __init__(
            self,
            finbert: ft.FineTunedFinBERT,
            hemlp: ModelBeijin,
            n_layers: int = 3,
            log_hparams: bool = True,
            **kwargs,
    ):
        """
        :param finbert:
        :param hemlp:
        :param n_layers: number of hidden layers of the classification MLP that is put
            on top of the finbert and hemlp models. This means that the total number of layers of such MLP
            is n_layer+2, as one input and output layers are also created.
        :param log_hparams:
        :param kwargs:
        """
        super().__init__()

        # NOTE: this call saves all the parameters passed to __init__ into self.hparams
        #   For this reason, do not delete the parameters even if they seem unused
        self.save_hyperparameters(ignore=["finbert", "hemlp"])

        # For both hemlp and finbert:
        #   1. Remove the classification heads, which is functionally the same as
        #   making them Identity layers that do nothing. This is because we want to put
        #   another classification head on top, meaning that we are actually interested in
        #   "cutting" each model at their last non-classification layer
        #   2. Freeze the model, we just want to train the aggr. layers
        old_class_ft: CustomLoRA = finbert.model.classifier
        finbert.model.classifier = nn.Identity()
        self.finbert: nn.Module = finbert
        self.finbert.requires_grad_(False)

        # Need to traverse the modules (except last one) of the hemlp to access and replace the final layer
        linear_layer_names = [name for name, mod in list(hemlp.named_modules()) if "linear" in name]
        # Split layer name by dots to identify each module
        last_linear_layer_components = linear_layer_names[-1].split('.')
        module = hemlp
        for component in last_linear_layer_components[:-1]:
            module = getattr(module, component)

        old_class_hemlp: nn.Linear = getattr(module, last_linear_layer_components[-1])
        setattr(module, last_linear_layer_components[-1], nn.Identity())
        self.hemlp: nn.Module = hemlp
        self.hemlp.requires_grad_(False)

        # Getting the in_features from the classification head is equal to the out_features of the last,
        #   non-classification layer
        in_features = old_class_ft.old_linear.in_features + old_class_hemlp.in_features
        self.aggregation_layers = nn.Sequential(
            collections.OrderedDict([
                ("linear_1", nn.Linear(in_features=in_features, out_features=EndToEndModel._OUT_FEATURES)),
                *[
                    (f"linear_{i}", nn.Linear(
                        in_features=EndToEndModel._OUT_FEATURES,
                        out_features=EndToEndModel._OUT_FEATURES
                    )) for i in range(1, self.hparams.n_layers+1)
                ],

                # Sentiment head: need 1 number that is crunched \in [-1, 1] by tanh
                ("sentiment", nn.Linear(in_features=EndToEndModel._OUT_FEATURES, out_features=1)),
                nn.Tanh()
            ])
        )

        # NOTE: need to manually call log_params here because mlflow.pytorch.autolog() doesn't log them
        #   and I am not using MLFlowLogger because it apparently duplicates logging when autolog() is active
        # NOTE2: synchronous false because this hangs if log_params has already been called for the current run
        #   I would like it to fail silently instead of blocking my training until it reaches the timeout
        if log_hparams:
            mlflow.log_params(self.hparams, synchronous=False)

    def forward(self, batch) -> torch.Tensor:
        tokenizer_output, embeddings, new_features = batch
        finbert_out = self.finbert(**tokenizer_output)
        hemlp_out = self.hemlp(embeddings, new_features)

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
        return self.aggregation_layers.state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # The state dict is just that of the aggr layers, since everything else is frozen
        return self.aggregation_layers.load_state_dict(state_dict, strict=strict, assign=assign)
