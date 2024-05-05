import json
import os
import typing

import training.loader as loader

MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'
"""
Host and port that the tracking server is started with in mlflow_server.sh
"""

BEST_REGISTERED_MODEL_ALIAS = 'champion'

EVALUATION_EXPERIMENT_NAME = 'Evaluation'


def get_experiment_name_prefix():
    return os.environ['EXPERIMENT_NAME_PREFIX']


def set_experiment_name_prefix(prefix: str):
    os.environ['EXPERIMENT_NAME_PREFIX'] = prefix


def get_model_choice() -> loader.Model:
    return loader.Model(os.environ.get('MODEL_CHOICE', loader.Model.FINBERT.value))


def set_model_choice(model_choice: loader.Model):
    os.environ['MODEL_CHOICE'] = model_choice.value


def get_dataset_choice() -> loader.Dataset:
    return loader.Dataset(os.environ.get('DATASET_CHOICE', loader.Dataset.SEMEVAL_TRAIN_VAL.value))


def set_dataset_choice(dataset_choice: loader.Dataset):
    os.environ['DATASET_CHOICE'] = dataset_choice.value


def get_experiment_name():
    return f"{get_experiment_name_prefix()}|{get_dataset_choice().value}|{get_model_choice().value}"


def set_registered_model_name_prefix(prefix: str):
    os.environ['REGISTERED_MODEL_NAME_PREFIX'] = prefix


def get_registered_model_name_prefix() -> str:
    return os.environ.get('REGISTERED_MODEL_NAME_PREFIX', 'best')


def get_registered_model_name(model_choice: loader.Model):
    return f"{get_registered_model_name_prefix()}-{model_choice.value}"


# Kind of dirty way of setting tags for every run
def set_run_tags(tags: typing.Dict[str, str | int]):
    os.environ['RUN_TAGS'] = json.dumps(tags)


def get_run_tags() -> typing.Dict[str, str | int]:
    return json.loads(os.environ.get('RUN_TAGS', '{}'))


def set_common_run_tags(
        with_neutral_samples: bool = False,
        fine_tuning_on_sc_dataset: bool = False
):
    tags = get_run_tags()

    if with_neutral_samples:
        tags['neutral_samples'] = 'true'

    if fine_tuning_on_sc_dataset:
        tags['fine_tuning_on_sc_dataset'] = 'true'

    set_run_tags(tags)
