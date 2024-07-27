import os

import click
import datasets
import pyspark.sql as psql
from loguru import logger

import data.semeval_2017_dataset as sem
import data.spark as S
import hand_eng_mlp_TODO.datasets.preprocessing_base as ppb
import utils.io as io_
from utils.random import RND_SEED

_MODEL_NAME = 'hand_engineered_mlp'
_DATASET_NAME = 'semeval2017'
_SPARK_APP_NAME = f'{_MODEL_NAME}|{_DATASET_NAME} Preprocessing'


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
def _main():
    raw_df_path = sem.download_dataset()  # get combined dataset path

    spark = S.create_spark_session(
        app_name=_SPARK_APP_NAME,
    )
    raw_df = sem.read_dataset(spark=spark, path=raw_df_path)
    # raw_df = raw_df.limit(100)  # Take the first 100 rows TODO: Remove later, just for testing

    train_ratio = 0.8
    test_ratio = 0.2
    raw_df_train, raw_df_test = raw_df.randomSplit([train_ratio, test_ratio], seed=RND_SEED)

    preprocess_and_save(spark=spark,
                        raw_df=raw_df_train,
                        is_train=True)
    preprocess_and_save(spark=spark,
                        raw_df=raw_df_test,
                        is_train=False)


def preprocess_and_save(
        spark: psql.SparkSession,
        raw_df: psql.DataFrame,
        is_train: bool):
    dataset_type = "train" if is_train else "test"
    logger.info(f"Cleaning {dataset_type} data...")
    df = sem.clean_dataset(df=raw_df)

    df = df.withColumnRenamed(sem.SENTIMENT_SCORE_COL, ppb.LABEL_COL)
    df = ppb.preprocess_dataset(
        spark=spark,
        raw_df=df,
        drop_neutral_samples=False,  # NOTE: false in this case because labels are continuously-valued sentiment scores
        text_col=sem.TEXT_COL,
        label_col=ppb.LABEL_COL
    )

    dataset_path = TRAIN_DATASET_PATH if is_train else VAL_DATASET_PATH
    logger.info(f"Preprocessing {dataset_type} dataset...")
    df.write.parquet(str(dataset_path), mode='overwrite')
    logger.info(f"Preprocessing {dataset_type} finished")


if __name__ == "__main__":
    _main()
