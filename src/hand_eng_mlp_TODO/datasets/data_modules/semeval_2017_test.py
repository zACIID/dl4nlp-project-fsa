import datasets
import lightning as L
import torch
from torch.utils.data import DataLoader

import hand_eng_mlp_TODO.datasets.preprocessing_base as ppb
import hand_eng_mlp_TODO.datasets.preprocessing_features_extraction as ppf
import hand_eng_mlp_TODO.datasets.semeval_2017.preprocessing as sem_pp
from data.semeval_2017_dataset import TEXT_COL
from utils.random import RND_SEED


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class Semeval2017Test(L.LightningDataModule):
    def __init__(
            self,
            test_batch_size: int = 32,
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
        :param with_neutral_samples: whether to load the dataset containing neutrally-labelled samples
        :param kwargs:
        """

        super().__init__()

        self.dataset: datasets.Dataset = sem_pp.get_dataset(train_dataset=False)
        self.test_batch_size = test_batch_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.rnd_seed = rnd_seed

    def prepare_data(self):
        # Nothing to do here since the dataset is provided from the outside
        pass

    def setup(self, stage: str = None):
        self.dataset.set_format(type='torch', columns=[ppb.EMBEDDER_OUTPUT_COL, *ppf.NEW_FEATURES, ppb.LABEL_COL])

    def train_dataloader(self):
        raise NotImplementedError("This data module is only for test datasets")

    def val_dataloader(self):
        raise NotImplementedError("This data module is only for test datasets")

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.test_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            collate_fn=_collate_fn
        )

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("This data module is only for test datasets")


def _collate_fn(raw_samples):
    embeddings = torch.nn.utils.rnn.pad_sequence(
        [item[ppb.EMBEDDER_OUTPUT_COL] for item in raw_samples],
        batch_first=True
    )

    scores = torch.tensor([item[ppb.LABEL_COL] for item in raw_samples])
    new_features = torch.stack([torch.tensor([item[key] for key in ppf.NEW_FEATURES]) for item in raw_samples])

    # print("embeddings batch ", embeddings_tensor.shape)
    # print("features batch ", new_features_tensor.shape)
    # print("scores batch ", scores.shape)

    return embeddings, scores, new_features

