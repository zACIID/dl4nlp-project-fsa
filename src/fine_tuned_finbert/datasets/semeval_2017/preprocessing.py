import os

import click
import datasets
from loguru import logger

import fine_tuned_finbert.datasets.preprocessing_base as ppb
import data.spark as S
import data.semeval_2017_dataset as sem
import fine_tuned_finbert.models.fine_tuned_finbert as ft
import utils.io as io_

_MODEL_NAME = 'finbert'
_DATASET_NAME = 'semeval2017'
_SPARK_APP_NAME = f'{_MODEL_NAME}|{_DATASET_NAME} Preprocessing'

_TOKENIZER_PATH = ft.PRE_TRAINED_MODEL_PATH

TRAIN_DATASET_PATH = io_.DATA_DIR / f'{_DATASET_NAME}-{_MODEL_NAME}-val.parquet'
VAL_DATASET_PATH = io_.DATA_DIR / f'{_DATASET_NAME}-{_MODEL_NAME}-test.parquet'


def get_dataset(train_dataset: bool) -> datasets.Dataset:
    """
    :param train_dataset: if True, returns train dataset, else test dataset
    :return:
    """

    dataset_path = TRAIN_DATASET_PATH if train_dataset else VAL_DATASET_PATH
    if not os.path.exists(dataset_path):
        raise FileNotFoundError('Dataset not found. Make sure to run this script to execute '
                                'the preprocessing pipeline for this dataset')

    # Have to load the actual .parquet files inside the dataset folder
    return datasets.Dataset.from_parquet(str(dataset_path / "*.parquet"))


@click.command(
    help=f"Preprocess {_MODEL_NAME} dataset"
)
@click.option("--get-train-dataset", '-d', is_flag=True, type=click.BOOL)
def _main(get_train_dataset: bool):
    raw_df_path = sem.download_dataset(return_train_dataset=get_train_dataset)

    spark = S.create_spark_session(
        app_name=_SPARK_APP_NAME,
    )
    raw_df = sem.read_dataset(spark=spark, path=raw_df_path)

    logger.info("Cleaning data...")
    df = sem.clean_dataset(
        df=raw_df,
    )

    df = df.withColumnRenamed(sem.SENTIMENT_SCORE_COL, ppb.LABEL_COL)

    df = ppb.preprocess_dataset(
        raw_df=df,
        text_col=sem.TEXT_COL,
    )

    # TODO ( ͡° ͜ʖ ͡°) need to derive a test dataset from the training dataset
    #   maybe merge 'Microblog_Trainingdata.json' and 'Microblog_Trialdata.json'
    #   and then take 200 samples out to make the test set, remaining samples for training
    dataset_path = TRAIN_DATASET_PATH if get_train_dataset else VAL_DATASET_PATH
    logger.info("Preprocessing dataset...")
    df.write.parquet(str(dataset_path), mode='overwrite')
    logger.info("Preprocessing finished")


if __name__ == "__main__":
    _main()
