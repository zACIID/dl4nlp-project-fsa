import os

import dotenv

import training.loader as loader
import utils.io as io_

dotenv.load_dotenv(str(io_.PROJECT_ROOT / 'mlflow.env'))

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MLFLOW_EXPERIMENT_NAME_PREFIX = 'Hyperopt'
MODEL_CHOICE = loader.Model.FINBERT
# DATASET_CHOICE = loader.Dataset.SEMEVAL_TRAIN_VAL
DATASET_CHOICE = loader.Dataset.SC_TRAIN_SEMEVAL_VAL
MLFLOW_EXPERIMENT_NAME = f"{MLFLOW_EXPERIMENT_NAME_PREFIX}|{DATASET_CHOICE}|{MODEL_CHOICE}"

BEST_REGISTERED_MODEL_ALIAS = 'champion'
MODEL_CHOICE_TAG = 'model_choice'
