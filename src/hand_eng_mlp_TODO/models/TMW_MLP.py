import typing
import warnings
from typing import Any, Type

import lightning as L
import mlflow
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.fine_tuned_finbert.models.loss_functions import sign_accuracy_mask
from src.hand_eng_mlp_TODO.models.super_MLP.base.base_MLP import BaseSuperMLP


class TMW_MLP(L.LightningModule):
    def __init__(
            self, model_type: Type[BaseSuperMLP], model_args: dict[str, Any],
            one_cycle_max_lr: float = 2e-5, weight_decay: float = 0.0,
            one_cycle_pct_start: float = 0.3, log_hparams: bool = True, **kwargs
    ) -> None:

        """
        :param model_type: defining which subclass of a BaseSuperMLP will be used
        :param model_args: arguments of the related SuperMLP
        :param one_cycle_max_lr: maximum learning rate that the LR scheduler will
              push the optimizer to
        :param weight_decay: defining how much we want the decay to affect the loss
        :param one_cycle_pct_start: The percentage of the cycle (in number of steps)
              spent increasing the learning rate
        """

        super().__init__()

        self.save_hyperparameters()

        if log_hparams:
            mlflow.log_params(self.hparams, synchronous=False)

        self.model: BaseSuperMLP = model_type(**model_args)
        self._val_predictions: list[Tensor] = []
        self._val_targets: list[Tensor] = []

    def forward(self, x_batch: Tensor) -> Tensor:
        return self.model(x_batch)

    def predict(self, x_batch: Tensor) -> Tensor:
        self.eval()

        with torch.no_grad():
            y_pred: Tensor = self(x_batch)
            return y_pred

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        if len(args) > 0:
            warnings.warn(f"Args are ignored by {self.__class__.__name__}, "
                          f"make sure to pass a BaseSuperMLP dictionary")

        return self.predict(**kwargs)

    def training_step(
            self, batch: tuple[Tensor, Tensor],
            batch_idx: int, dataloader_idx: int = 0) -> Tensor:

        return self._base_step(batch, batch_idx, dataloader_idx, step_type="train")

    def validation_step(
            self, batch: tuple[Tensor, Tensor],
            batch_idx: int, dataloader_idx: int = 0) -> Tensor:

        return self._base_step(batch, batch_idx, dataloader_idx, step_type="val")

    def test_step(
            self, batch: tuple[Tensor, Tensor],
            batch_idx: int, dataloader_idx: int = 0) -> Tensor:

        return self._base_step(batch, batch_idx, dataloader_idx, step_type="test")

    def base_step(
            self, batch: tuple[Tensor, Tensor],
            batch_idx: int, dataloader_idx: int = 0, step_type: str = None) -> Tensor:

        x_batch, y_batch = batch
        y_pred: Tensor = self.forward(x_batch)

        mse: Tensor = F.mse_loss(y_batch, y_pred)
        mae: Tensor = F.l1_loss(y_batch, y_pred)

        batch_len: int = len(y_batch) if len(y_batch) > 0 else 1

        sign_acc: Tensor = torch.sum(
            sign_accuracy_mask(y_batch, y_pred)
        ) / batch_len

        # TODO check if cosine similarity is a good loss for our MLP
        loss: Tensor = -torch.cosine_similarity(y_batch, y_pred, dim=-1)

        if step_type == 'val':
            self._val_predictions.append(y_pred)
            self._val_targets.append(batch[1])

        self.log_dict(
            dictionary={
                f"{step_type}_loss": loss,
                f"{step_type}_mse": mse,
                f"{step_type}_mae": mae,
                f"{step_type}_sign_accuracy": sign_acc,
                f"{step_type}_positive_predictions": sign_accuracy_mask(
                    torch.ones(y_batch.shape, device=self.device), y_pred
                ).sum() / batch_len,
                f"{step_type}_mean_prediction": y_pred.mean(),
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

        cosine_similarity: Tensor = F.cosine_similarity(predictions, targets, dim=0)
        self.log(
            name=f"val_cosine_similarity",
            value=cosine_similarity,
            on_step=False,  # True raises error because we are on epoch end
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

    def configure_optimizers(self) -> tuple[list, list]:
        no_decay = ["bias", "layernorm"]
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

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=1e-3,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.one_cycle_max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.one_cycle_pct_start
        )

        scheduler = {
            "name": OneCycleLR.__name__,
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return [optimizer], [scheduler]
