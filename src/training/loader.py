import collections
import typing

import mlflow
from lightning import LightningModule, LightningDataModule
from mlflow.entities.model_registry import ModelVersion

import fine_tuned_finbert.datasets.data_modules as ft_dm
import hand_eng_mlp_TODO.datasets.data_modules as mlp_dm
import end_to_end_TODO.data_modules as e2e_dm
import utils.mlflow_env as env
from end_to_end_TODO.models.end_to_end_model import EndToEndModel
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
            return _load_end_to_end_model_and_data_module(model_init_args, dataset_choice, dm_init_args)
        case _:
            raise ValueError(f'Unknown model {model_choice}')


def _load_finbert_model_and_data_module(
        model_init_args: typing.MutableMapping[str, typing.Any],
        dataset_choice: Dataset,
        dm_init_args: typing.Mapping[str, typing.Any]
) -> typing.Tuple[LightningModule, LightningDataModule]:
    # TODO this is janky at best, because it relies on the fact that a) hyperopt is the only place that
    #   instantiates the pretrained models for training and b) that the search space, i.e. args passed,
    #   is defined in such a way that only non-architecture-altering params are provided
    if env.should_hyperopt_on_pretrained_model():
        model = load_best_model(
            Model.FINBERT,
            load_pretrained_on_sc=True,
            init_kwargs=collections.ChainMap(
            model_init_args,
            {
                "strict": False,
                "log_hparams": False
            }
        ))
    else:
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
    # TODO NOTE: now providing a default so that semeval_full_training doesn't break
    model_init_args.setdefault("model_spec", MLPType.BOX.value)

    # TODO this is janky at best, because it relies on the fact that a) hyperopt is the only place that
    #   instantiates the pretrained models for training and b) that the search space, i.e. args passed,
    #   is defined in such a way that only non-architecture-altering params are provided
    if env.should_hyperopt_on_pretrained_model():
        model = load_best_model(
            Model.HAND_ENG_MLP,
            load_pretrained_on_sc=True,
            init_kwargs=collections.ChainMap(
                model_init_args,
                {
                    "strict": False,
                    "log_hparams": False
                }
            ))
    else:
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
        if model_init_args["model_spec"] == 1:
            model_init_args["model_spec"] = MLPType.BOX
        elif model_init_args["model_spec"] == 2:
            model_init_args["model_spec"] = MLPType.RHOMBOID
            data["beta"] = model_init_args.pop("beta", 0.5)
        elif model_init_args["model_spec"] == 3:
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


def _load_end_to_end_model_and_data_module(
        model_init_args: typing.MutableMapping[str, typing.Any],
        dataset_choice: Dataset,
        dm_init_args: typing.Mapping[str, typing.Any]
) -> typing.Tuple[LightningModule, LightningDataModule]:
    # Load the best finbert and mlp models for the e2e
    model_init_args["finbert"] = load_best_model(Model.FINBERT)
    model_init_args["hemlp"] = load_best_model(Model.HAND_ENG_MLP)

    model = EndToEndModel(**model_init_args)
    match dataset_choice:
        case Dataset.SC_TRAIN_VAL:
            raise ValueError("Dataset not implemented for E2E model")
        case Dataset.SC_TRAIN_SEMEVAL_VAL:
            return model, e2e_dm.StocktwitsCryptoTrainSemEval2017Val(**dm_init_args)
        case Dataset.SEMEVAL_TRAIN:
            return model, e2e_dm.Semeval2017Train(**dm_init_args)
        case Dataset.SEMEVAL_TRAIN_VAL:
            return model, e2e_dm.Semeval2017TrainVal(**dm_init_args)
        case Dataset.SEMEVAL_TEST:
            return model, e2e_dm.Semeval2017Test(**dm_init_args)
        case _:
            raise ValueError(f'Unknown dataset {dataset_choice}')


def load_best_model(model: Model, load_pretrained_on_sc: bool = False, **init_kwargs) -> LightningModule:
    """
    Loads the best model for the provided model type from the mlflow registry.
    :param model: model type
    :param load_pretrained_on_sc: whether to load the best pretrained model on the SC dataset
        or the best fully-trained model
    :param init_kwargs: keyword arguments forwarded to the class method LightningModule.load_from_checkpoint,
        which itself forwards the ones that it doesn't use to initialize the model.
        TODO since this implementation is kinda dirty, some provided kwargs may be overridden, see the code
    :return:
    """
    model_name = env.get_registered_model_name(model)
    alias = env.get_dataset_specific_best_model_alias(dataset=Dataset.SC_TRAIN_SEMEVAL_VAL, tuning_alias=True) \
        if load_pretrained_on_sc \
        else env.BEST_FULL_TRAINED_MODEL_ALIAS

    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=env.MLFLOW_TRACKING_URI)
    best_version: ModelVersion = client.get_model_version_by_alias(name=model_name, alias=alias)

    mlflow.set_tag(key='model_name', value=model_name)
    mlflow.set_tag(key='model_alias', value=alias)
    mlflow.set_tag(key='model_version', value=best_version.version)

    match model:
        case Model.FINBERT:
            model_class = FineTunedFinBERT
        case Model.HAND_ENG_MLP:
            model_class = ModelBeijin
        case Model.END_TO_END:
            model_class = EndToEndModel
            init_kwargs["finbert"] = load_best_model(Model.FINBERT)
            init_kwargs["hemlp"] = load_best_model(Model.HAND_ENG_MLP)
        case _:
            raise ValueError(f'Unknown model {model}')

    # Load the checkpoint however it is - we do not save every parameter in checkpoints
    init_kwargs["strict"] = False
    init_kwargs["log_hparams"] = False

    # NOTE: check the implementation at https://mlflow.org/docs/latest/_modules/mlflow/pytorch.html#load_checkpoint
    #   it passes any kwargs of this method other than the model_class to the
    #   class method LightningModule.load_from_checkpoint
    best_model: LightningModule = mlflow.pytorch.load_checkpoint(
        model_class=model_class,
        run_id=best_version.run_id,
        kwargs=init_kwargs
    )
    return best_model
