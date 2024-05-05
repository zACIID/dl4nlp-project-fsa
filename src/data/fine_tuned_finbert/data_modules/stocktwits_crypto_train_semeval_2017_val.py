import datasets
import lightning as L
import torch
from torch.utils.data import DataLoader

import data.fine_tuned_finbert.preprocessing_base as ppb
import data.fine_tuned_finbert.semeval_2017.preprocessing as sem_pp
import data.fine_tuned_finbert.stocktwits_crypto.preprocessing as sc_pp
from utils.random import RND_SEED


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class StocktwitsCryptoTrainSemEval2017Val(L.LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 64,
            eval_batch_size: int = 32,
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

        self.train_dataset: datasets.Dataset = sc_pp.get_dataset(drop_neutral_samples=with_neutral_samples)
        self.val_dataset: datasets.Dataset = sem_pp.get_dataset(train_dataset=True)  # use train dataset as eval dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.rnd_seed = rnd_seed

    def prepare_data(self):
        # Nothing to do here since the dataset is provided from the outside
        pass

    def setup(self, stage: str = None):
        self.train_dataset.set_format(type='torch', columns=[ppb.TOKENIZER_OUTPUT_COL, ppb.LABEL_COL])
        self.val_dataset.set_format(type='torch', columns=[ppb.TOKENIZER_OUTPUT_COL, ppb.LABEL_COL])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            collate_fn=_train_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.eval_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
            collate_fn=_val_collate_fn
        )

    def test_dataloader(self):
        raise NotImplementedError("This data module is only for training and validation datasets")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("This data module is only for training and validation datasets")


def _collate_fn(raw_samples, sentiment_score_col: str, tokenizer_col: str):
    tokenizer_outputs = [item[tokenizer_col] for item in raw_samples]
    scores = [item[sentiment_score_col] for item in raw_samples]

    input_ids = torch.stack(list(map(lambda x: x['input_ids'], tokenizer_outputs)))
    att_masks = torch.stack(list(map(lambda x: x['attention_mask'], tokenizer_outputs)))
    tensorized_tokenizer_output = {'input_ids': input_ids, 'attention_mask': att_masks}

    scores = torch.tensor(scores)

    return tensorized_tokenizer_output, scores


def _train_collate_fn(raw_samples):
    return _collate_fn(raw_samples, ppb.LABEL_COL, ppb.TOKENIZER_OUTPUT_COL)


def _val_collate_fn(raw_samples):
    return _collate_fn(raw_samples, ppb.LABEL_COL, ppb.TOKENIZER_OUTPUT_COL)
