import typing

import datasets
import lightning as L
import numpy as np
import sklearn.model_selection as sel
import torch
from torch.utils.data import DataLoader, Subset

import data.hand_engineered_mlp.preprocessing as pp
from utils.random import RND_SEED


class EndToEndTrainVal(L.LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 64,
            eval_batch_size: int = 8,
            train_split_size: float = 0.8,
            pin_memory: bool = False,
            prefetch_factor: int = 4,
            num_workers: int = 0,
            rnd_seed: int = RND_SEED,
            **kwargs,
    ):
        """
        :param dataset:
        :param train_batch_size:
        :param eval_batch_size: val/test/predict batch size
        :param train_split_size: fraction of data used for training.
            The remaining fraction of data will be used for validation
        :param kwargs:
        """

        super().__init__()

        self.dataset: datasets.Dataset = pp.get_dataset()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.rnd_seed = rnd_seed
        self.train_idxs: np.ndarray | None = None
        self.val_idxs: np.ndarray | None = None

    def prepare_data(self):
        # Nothing to do here since the dataset is provided from the outside
        pass

    def setup(self, stage: str = None):
        # TODO example
        # self.dataset.set_format(type='torch', columns=[pp.TOKENIZER_OUTPUT_COL, pp.SENTIMENT_SCORE_COL])
        # index = np.arange(len(self.dataset))
        # train_split_idxs, val_split_idxs = sel.train_test_split(
        #     index,
        #     stratify=self.dataset.with_format(type='pandas')[pp.SENTIMENT_SCORE_COL].to_numpy(),
        #     random_state=self.rnd_seed
        # )
        #
        # self.train_idxs = train_split_idxs
        # self.val_idxs = val_split_idxs
        raise NotImplementedError('TODO') # TODO

    def train_dataloader(self):
        return DataLoader(
            dataset=Subset(self.dataset, self.train_idxs),
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            collate_fn=_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=Subset(self.dataset, self.val_idxs),
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
            collate_fn=_collate_fn
        )

    def test_dataloader(self):
        raise NotImplementedError("This data module is only for training and validation datasets")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("This data module is only for training and validation datasets")


def _collate_fn(raw_samples: typing.List[typing.Any]):
    # TODO example - the purpose of this function is transform each row of the dataset into an actual batch
    #   notice how below a tuple of tensors is returned, each of which is a batch of model inputs and labels basically
    # tokenizer_outputs = [item[pp.TOKENIZER_OUTPUT_COL] for item in raw_samples]
    # scores = [item[pp.SENTIMENT_SCORE_COL] for item in raw_samples]
    #
    # input_ids = torch.stack(list(map(lambda x: x['input_ids'], tokenizer_outputs)))
    # att_masks = torch.stack(list(map(lambda x: x['attention_mask'], tokenizer_outputs)))
    # tensorized_tokenizer_output = {'input_ids': input_ids, 'attention_mask': att_masks}
    #
    # scores = torch.tensor(scores)
    #
    # return tensorized_tokenizer_output, scores
    raise NotImplementedError('TODO') # TODO
