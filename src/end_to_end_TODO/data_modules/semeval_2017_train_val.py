import datasets
import lightning as L
import numpy as np
import sklearn.model_selection as sel
import torch
from torch.utils.data import DataLoader, Subset

import fine_tuned_finbert.datasets.preprocessing_base as ppb_ft
import fine_tuned_finbert.datasets.semeval_2017.preprocessing as sem_pp_ft
import hand_eng_mlp_TODO.datasets.preprocessing_base as ppb_mlp
import hand_eng_mlp_TODO.datasets.preprocessing_features_extraction as ppfe_mlp
import hand_eng_mlp_TODO.datasets.semeval_2017.preprocessing as sem_pp_mlp
import fine_tuned_finbert.datasets.data_modules as ft_dm
import hand_eng_mlp_TODO.datasets.data_modules as mlp_dm
from utils.random import RND_SEED


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class Semeval2017TrainVal(L.LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            train_split_size: float = 0.9,
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
        :param kwargs:
        """

        super().__init__()

        self._finbert_dataset: datasets.Dataset = sem_pp_ft.get_dataset(train_dataset=True)
        # cannot have duplicate cols when concatenating datasets
        self._hemlp_dataset: datasets.Dataset = (sem_pp_mlp.get_dataset(train_dataset=True)
                                                 .remove_columns(column_names=["id", "spans", "label"]))

        # Concatenate the two datasets horizontally
        # Since they refer to the same data, they should have the same number of rows
        self.dataset = datasets.concatenate_datasets([self._finbert_dataset, self._hemlp_dataset], axis=1)

        self.train_split_size = train_split_size
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
        self.dataset.set_format(
            type='torch',
            columns=[
                ppb_ft.TOKENIZER_OUTPUT_COL,
                ppb_ft.LABEL_COL,
                ppb_mlp.EMBEDDER_OUTPUT_COL,
                *ppfe_mlp.NEW_FEATURES
            ]
        )
        index = np.arange(len(self.dataset))
        train_split_idxs, val_split_idxs = sel.train_test_split(
            index,
            train_size=self.train_split_size,
            stratify=(self.dataset.with_format(type='pandas')[ppb_ft.LABEL_COL].to_numpy() >= 0).astype(int),
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


_FT_DM = ft_dm.Semeval2017TrainVal()
_FT_DM.setup()
_FT_COLLATE_FN = _FT_DM.train_dataloader().collate_fn
_HEMLP_DM = mlp_dm.Semeval2017TrainVal()
_HEMLP_DM.setup()
_HEMLP_COLLATE_FN = _HEMLP_DM.train_dataloader().collate_fn
def _collate_fn(raw_samples):
    # collate_fn is the same for any instance of the same datamodule (i.e. regardless of init params)
    #   and every dataloader
    tokenizer_output, scores = _FT_COLLATE_FN(raw_samples)
    embeddings, _, new_features = _HEMLP_COLLATE_FN(raw_samples)

    # TODO future refactorings:
    # - This means that I would have to extract the collate_fn functions into a single file, for both finbert and hemlp
    #   and then reference them here. I Could also define this collate_fn into a separate file
    #   Maybe such file could be datamodules/collate.py
    # - I am also not sure about the fact of putting datamodules/ inside datasets/, maybe
    #   put them at the <model_root> level? Also they could be refactored because all they do is just hardcode the dataset to fetch,
    #   the rest is code repetition...
    return tokenizer_output, embeddings, new_features, scores

