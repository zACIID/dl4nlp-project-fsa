import enum
import typing

from lightning import LightningModule, LightningDataModule

import data.fine_tuned_finbert.data_modules as ft_dm
from models.fine_tuned_finbert import FineTunedFinBERT


class Dataset(enum.StrEnum):
    SC_TRAIN_VAL = 'SC_TRAIN_VAL'
    SC_TRAIN_SEMEVAL_VAL = 'SC_TRAIN_SEMEVAL_VAL'
    SEMEVAL_TRAIN_VAL = 'SEMEVAL_TRAIN_VAL'
    SEMEVAL_TEST = 'SEMEVAL_TEST'


class Model(enum.StrEnum):
    FINBERT = 'FINBERT'
    HAND_ENG_MLP = 'HAND_ENG_MLP'
    END_TO_END = 'END_TO_END'


def get_model_and_data_module(
        model_choice: Model,
        model_init_args: typing.Dict[str, typing.Any],
        dataset_choice: Dataset,
        dm_init_args: typing.Dict[str, typing.Any]
) -> typing.Tuple[LightningModule, LightningDataModule]:
    match model_choice:
        case Model.FINBERT:
            return load_finbert_model_and_data_module(model_init_args, dataset_choice, dm_init_args)
        case Model.HAND_ENG_MLP:
            raise NotImplementedError()  # TODO implement same function as above
        case Model.END_TO_END:
            raise NotImplementedError()  # TODO implement same function as above
        case _:
            raise ValueError(f'Unknown model {model_choice}')


def load_finbert_model_and_data_module(
        model_init_args: typing.Dict[str, typing.Any],
        dataset_choice: Dataset,
        dm_init_args: typing.Dict[str, typing.Any]
) -> typing.Tuple[LightningModule, LightningDataModule]:
    model = FineTunedFinBERT(**model_init_args)
    match dataset_choice:
        case Dataset.SC_TRAIN_VAL:
            return model, ft_dm.StocktwitsCryptoTrainVal(**dm_init_args)
        case Dataset.SC_TRAIN_SEMEVAL_VAL:
            return model, ft_dm.StocktwitsCryptoTrainSemEval2017Val(**dm_init_args)
        case Dataset.SEMEVAL_TRAIN_VAL:
            return model, ft_dm.Semeval2017TrainVal(**dm_init_args)
        case Dataset.SEMEVAL_TEST:
            return model, ft_dm.SemEval2017Test(**dm_init_args)
        case _:
            raise ValueError(f'Unknown dataset {dataset_choice}')
