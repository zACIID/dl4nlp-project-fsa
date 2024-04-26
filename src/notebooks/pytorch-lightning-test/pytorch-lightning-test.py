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
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PyTorch Lightning Tests
#
# Stuff that I would like to do:
# - [ ] implement LightningModule
# - [ ] use torchmetrics to calculate losses and stuff
# - [ ] setup and connect mlflow to lightning modules
#     - made in such a way that the mlflow tracking server is easily hot-swappable with a managed server such as one hosted by databricks; this way we can move to colab if necessary and still use mlflow
# - [ ] log metrics to mlflow
# - [ ] learning rate scheduling
# - [ ] early stopping
#     - see https://lightning.ai/docs/pytorch/stable/common/early_stopping.html#early-stopping   
# - [ ] checkpoint best models and log them to mlflow
#     - See [here](https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling) to customize checkpointing (i.e. when to create the model, based on what metric improvement etc.)
# - [ ] gradient checkpointing and accumulation
#     - *gradient accumulation*: https://lightning.ai/docs/pytorch/stable/common/trainer.html#accumulate-grad-batches
#         - at least a basic version, see docs linked in the above section if more detail is needed  
# - [ ] mixed precision
#     - this too is a Trainer parameter, in my case the interesting one would be "16-mixed"  
# - [ ] LoRA
# - [ ] Retrieve a model from the artifacts stored in mlflow, the best model ideally, and then load it to train/make predictions 
#
# *questions for myself*:
# - do we need to manually add the CLS and SEP tokens at the end of each sentence or the tokenizer does that already, if needed?
#
# *cool stuff*:
# - `fast_dev_run` and `overfit_batches` Trainer parameters to easily check if the model runs without errors

# %%
import os
from typing import Any, Union, Optional, Callable

import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import lightning as L
# import torchmetrics as tm # TODO consider using this if particular metrics are needed

# %% [markdown]
# ## Lit Wrapper
#

# %%
class LitWrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        
        self.model: nn.Module = model
        
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # TODO idk if need to override
        super().forward()
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # training_step defines the train loop.
        x = batch  # x is a dictionary with input_ids, attention_mask, etc.
        preds = self.model(**x)
        loss = F.cross_entropy(preds.logits, x["input_ids"])
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True) 
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        # NOTE: gradient calculation is disabled under the hood,
        #   see https://github.com/Lightning-AI/pytorch-lightning/blob/10c643f162318b7fe2b4a041a1f2975468492a92/pytorch_lightning/trainer/evaluation_loop.py#L246
        # seealso https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html#validate-and-test-a-model-basic
        # seealso here to define after how many training epochs to run a validation epoch https://lightning.ai/docs/pytorch/stable/advanced/speed.html#control-validation-frequency

        x = batch  # x is a dictionary with input_ids, attention_mask, etc.
        preds = self.model(**x)
        loss = F.cross_entropy(preds.logits, x["input_ids"])
        self.log("val_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        # NOTE: grad is disabled by Trainer automatically, see https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html#validate-and-test-a-model-basic
        
        x = batch  # x is a dictionary with input_ids, attention_mask, etc.
        preds = self.model(**x)
        loss = F.cross_entropy(preds.logits, x["input_ids"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        # TODO what would I need this for? 
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def lr_scheduler_step(self, scheduler, metric):
        # Needed with non-PyTorch compliant schedulers, see
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html#bring-your-own-custom-learning-rate-schedulers
        # But shoudl I call this manually? see here https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling
        #   As of now not clear when this method is called and how it interacts with the optimizers
        raise NotImplementedError()
    
    
class LitDataModule(L.LightningDataModule):
    # TODO check the use of the following methods here https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py
    # TODO the above example is very useful in our case I think because it involves a pre-trained BERT
    
    # This class is then fed as arg to Trainer.fit
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError()
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()
    
    def setup(self, stage: str) -> None:
        # apparently this is used to create the dataset
        raise NotImplementedError()
    
    def prepare_data(self) -> None:
        # used to load tokenizer and dataset? need to understand the difference from setup()
        raise NotImplementedError()
    
    def convert_to_features(self):
        # TODO this is not defined in the LightningModule interface, but it is used in the aforementioned
        #   example to convert the corpus (dataset) to tokens
        raise NotImplementedError()

# %%
# TODO
# 1. load dataset, see https://huggingface.co/docs/datasets/v1.13.0/use_dataset.html to understand
#   how to transform tokenizer output into a dataset suitable for a dataloader
# 2. load model and convert to cuda if possible
#   1. This is apparently done with the Trainer class, which also handles moving stuff to the correct devices: https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
# 3. create dataloader
# 4. create Trainer instance and start fitting
#   1. see https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html#train-the-model
#   2. follow this to set seed and enable deterministic training https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html#validate-and-test-a-model-basic
#   3. Trainer(enable_progress_bar=True)
# 5. Validation and test loop as seen here:
#   https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html#validate-and-test-a-model-basic
#   1. do I pass train and val dataloaders to Trainer.fit or do I use Trainer.validate as shown here? https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html#validate-and-test-a-model-basic
# 6. Additional logging via callbacks passed to the Trainer: https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling


# TODO IMPORTANT: check this https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html#example-bert-nlp to see how to combine pre=trained bert and pytorch lightning
