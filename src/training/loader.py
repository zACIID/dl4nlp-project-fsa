import enum
import typing

from lightning import LightningModule, LightningDataModule

import fine_tuned_finbert.datasets.data_modules as ft_dm
import hand_eng_mlp_TODO.datasets.data_modules as mlp_dm
from fine_tuned_finbert.models.fine_tuned_finbert import FineTunedFinBERT

from hand_eng_mlp_TODO.models.model_beijin import ModelBeijin, MLPType
from utils.loader_types import Model, Dataset


def get_model_and_data_module(
        model_choice: Model,
        model_init_args: typing.MutableMapping[str, typing.Any],
        dataset_choice: Dataset,
        dm_init_args: typing.Mapping[str, typing.Any]
) -> typing.Tuple[LightningModule, LightningDataModule]:
    match model_choice:
        case Model.FINBERT:
            return _load_finbert_model_and_data_module(model_init_args, dataset_choice, dm_init_args)
        case Model.HAND_ENG_MLP:
            return _load_model_beijin_and_data_module(model_init_args, dataset_choice, dm_init_args)
        case Model.END_TO_END:
            raise NotImplementedError() # TODO ( ͡° ͜ʖ ͡°) implement same function as above
        case _:
            raise ValueError(f'Unknown model {model_choice}')


def _load_finbert_model_and_data_module(
        model_init_args: typing.MutableMapping[str, typing.Any],
        dataset_choice: Dataset,
        dm_init_args: typing.Mapping[str, typing.Any]
) -> typing.Tuple[LightningModule, LightningDataModule]:
    model = FineTunedFinBERT(**model_init_args)
    match dataset_choice:
        case Dataset.SC_TRAIN_VAL:
            return model, ft_dm.StocktwitsCryptoTrainVal(**dm_init_args)
        case Dataset.SC_TRAIN_SEMEVAL_VAL:
            return model, ft_dm.StocktwitsCryptoTrainSemEval2017Val(**dm_init_args)
        case Dataset.SEMEVAL_TRAIN:
            return model, ft_dm.Semeval2017Train(**dm_init_args)
        case Dataset.SEMEVAL_TRAIN_VAL:
            return model, ft_dm.Semeval2017TrainVal(**dm_init_args)
        case Dataset.SEMEVAL_TEST:
            return model, ft_dm.Semeval2017Test(**dm_init_args)
        case _:
            raise ValueError(f'Unknown dataset {dataset_choice}')


def _load_model_beijin_and_data_module(
        model_init_args: typing.MutableMapping[str, typing.Any],
        dataset_choice: Dataset,
        dm_init_args: typing.Mapping[str, typing.Any]
) -> typing.Tuple[LightningModule, LightningDataModule]:

    # NOTE: need to specify default values because the full-training script
    #   loads the model without passing any init keyword basically, the reason
    #   being that those from the best model are later used
    data: typing.Dict[str, typing.Any] = {
        "n_layers": model_init_args.pop("n_layers", 10),
        "dropout": model_init_args.pop("dropout", 0.2),
        "linear": model_init_args.pop("linear", False),
        "layernorm": model_init_args.pop("layernorm", True)
    }

    # The 0, 1, 2 here is not defined by the enum but by the hyperopt search space
    if model_init_args["model_spec"] == 0:
        model_init_args["model_spec"] = MLPType.BOX
    elif model_init_args["model_spec"] == 1:
        model_init_args["model_spec"] = MLPType.RHOMBOID
        data["beta"] = model_init_args.pop("beta", 0.5)
    elif model_init_args["model_spec"] == 2:
        model_init_args["model_spec"] = MLPType.REPRHOMBOID
        data["beta"] = model_init_args.pop("beta", 0.5)
    else:
        raise ValueError(f'Unknown model spec {model_init_args["model_spec"]}')

    model_init_args["MLP_args"] = data
    model = ModelBeijin(**model_init_args)

    match dataset_choice:
        case Dataset.SC_TRAIN_VAL:
            return model, mlp_dm.StocktwitsCryptoTrainVal(**dm_init_args)
        case Dataset.SC_TRAIN_SEMEVAL_VAL:
            return model, mlp_dm.StocktwitsCryptoTrainSemEval2017Val(**dm_init_args)
        case Dataset.SEMEVAL_TRAIN:
            return model, mlp_dm.Semeval2017Train(**dm_init_args)
        case Dataset.SEMEVAL_TRAIN_VAL:
            return model, mlp_dm.Semeval2017TrainVal(**dm_init_args)
        case Dataset.SEMEVAL_TEST:
            return model, mlp_dm.Semeval2017Test(**dm_init_args)
        case _:
            raise ValueError(f'Unknown dataset {dataset_choice}')
