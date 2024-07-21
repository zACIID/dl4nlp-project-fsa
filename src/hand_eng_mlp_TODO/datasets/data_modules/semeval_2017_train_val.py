import typing

import datasets
import lightning as L
import numpy as np
import sklearn.model_selection as sel
import torch
from torch.utils.data import DataLoader, Subset

import fine_tuned_finbert.datasets.preprocessing_base as ppb
import fine_tuned_finbert.datasets.semeval_2017.preprocessing as pp #todo vedere cosa dfa
import hand_eng_mlp_TODO.datasets.preprocessing_base as hemlp_pb
from utils.random import RND_SEED


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class Semeval2017TrainVal(L.LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            train_split_size: float = 0.9,
            with_neutral_samples: bool = True,
            pin_memory: bool = False,
            prefetch_factor: int = 4,
            num_workers: int = 4,
            rnd_seed: int = RND_SEED,
            **kwargs,
    ):
        """
        :param dataset:
        :param train_batch_size:
        :param eval_batch_size: val/test/predict batch size
        :param train_split_size: fraction of data used for training.
            The remaining fraction of data will be used for validation
        :param with_neutral_samples: whether to load the dataset containing neutrally-labelled samples
        :param kwargs:
        """

        super().__init__()

        self.dataset: datasets.Dataset = pp.get_dataset(train_dataset=True)
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
        self.dataset.set_format(type='torch', columns=[ppb.TOKENIZER_OUTPUT_COL, ppb.LABEL_COL])
        index = np.arange(len(self.dataset))
        train_split_idxs, val_split_idxs = sel.train_test_split(
            index,
            stratify=(self.dataset.with_format(type='pandas')[ppb.LABEL_COL].to_numpy() >= 0).astype(int),
            random_state=self.rnd_seed
        )

        self.train_idxs = train_split_idxs
        self.val_idxs = val_split_idxs

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
            batch_size=self.eval_batch_size,
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


def _collate_fn(raw_samples):
    # TODO DO THIS FOR ALL data_modules under hang_eng_mlp (stocktwits files left)
    # TODO check embeddings shape

    embeddings = [torch.stack(item[hemlp_pb.EMBEDDER_OUTPUT_COL], dim=0) for item in raw_samples]
    scores = [item[hemlp_pb.LABEL_COL] for item in raw_samples]
    new_features = [[item[key] for key in hemlp_pb.NEW_FEATURES] for item in raw_samples]

    embeddings_tensor = torch.stack(embeddings, dim=0)
    new_features_tensor = torch.stack(new_features)
    scores = torch.tensor(scores)

    print("embeddings batch ", embeddings_tensor.shape)
    print("features batch ", new_features_tensor.shape)
    print("scores batch ", scores.shape)

    return embeddings_tensor, scores, new_features_tensor

