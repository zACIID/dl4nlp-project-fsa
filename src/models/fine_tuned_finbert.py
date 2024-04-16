from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    BertForMaskedLM,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.tokenization_utils_base import BatchEncoding
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class FineTunedFinBERT(L.LightningModule):
    """
    Class that represents a model (currently "ahmedrachid/FinancialBERT" from HuggingFace) combined with
    all the fine-tuning infrastructure
    """

    def __init__(
            self,
            epochs: int,
            n_batches: int,
            model_name_or_path: str = "ahmedrachid/FinancialBERT",
            max_lr: float = 2e-5,
            weight_decay: float = 0.0,
            enable_gradient_checkpointing: bool = False,
            **kwargs,
    ):
        """
        :param model_name_or_path: pointing to local model or Huggingface hub
        :param loss_fun: loss function
        :param epochs: number of epochs the model will be trained for - needed by the LR scheduler
        :param n_batches: e.g. obtainable by `len(train_dataloader)` - needed by the LR scheduler
        :param max_lr: maximum learning rate that the LR scheduler will push the optimizer to
        :param weight_decay:
        :param kwargs:
        """
        super().__init__()

        # NOTE: this call saves all the parameters passed to __init__ into self.hparams
        #   For this reason, do not delete the parameters even if they seem unused
        self.save_hyperparameters(logger=True)

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                "use_reentrant": False
            })

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

    def forward(self, **inputs) -> MaskedLMOutput:
        return self.model(**inputs)

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
        outputs: MaskedLMOutput = self(**batch)

        # TODO change loss function
        # loss = F.cross_entropy(outputs.logits, labels)
        loss = torch.tensor([1.0], requires_grad=True)

        # TODO define other metrics to log, e.g. taken by torchmetrics or HuggingFace's evaluate package
        self.log_dict(
            dictionary={
                f"{step_type}_loss": loss,
            },
            on_step=True,
            on_epoch=True
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
            max_lr=self.hparams.max_lr,
            steps_per_epoch=self.hparams.n_batches,
            epochs=self.hparams.epochs,
        )

        # Docs on return values and scheduler config dictionary:
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        # As said above, OneCycleLR should be stepped after each optimizer step
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


