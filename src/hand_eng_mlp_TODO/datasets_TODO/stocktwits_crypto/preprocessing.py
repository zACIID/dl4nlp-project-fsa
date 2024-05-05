import os

import click
import datasets
from loguru import logger

import hand_eng_mlp.datasets_TODO.preprocessing_base as ppb
import data.spark as S
import data.stocktwits_crypto_dataset as sc
import utils.io as io_

_MODEL_NAME = 'hand_engineered_mlp'
_DATASET_NAME = 'stocktwits-crypto'
_SPARK_APP_NAME = f'{_MODEL_NAME}|{_DATASET_NAME} Preprocessing'

WITH_NEUTRALS_DATASET_PATH = io_.DATA_DIR / f'{_DATASET_NAME}-{_MODEL_NAME}-with-neutrals.parquet'
WITHOUT_NEUTRALS_DATASET_PATH = io_.DATA_DIR / f'{_DATASET_NAME}-{_MODEL_NAME}-without-neutrals.parquet'


def get_dataset(drop_neutral_samples: bool) -> datasets.Dataset:
    dataset_path = WITH_NEUTRALS_DATASET_PATH if drop_neutral_samples else WITHOUT_NEUTRALS_DATASET_PATH
    if not os.path.exists(dataset_path):
        raise FileNotFoundError('Dataset not found. Make sure to run this script to execute '
                                'the preprocessing pipeline for this dataset')

    # Have to load the actual .parquet files inside the dataset folder
    return datasets.Dataset.from_parquet(str(dataset_path / "*.parquet"))


@click.command(
    help=f"Preprocess {_MODEL_NAME} fine-tuning dataset"
)
@click.option("--drop-neutral-samples", '-d', is_flag=True, type=click.BOOL)
def _main(drop_neutral_samples: bool):
    raw_csv_path = sc.download_dataset()

    spark = S.create_spark_session(
        app_name=_SPARK_APP_NAME,
    )
    raw_df = sc.read_dataset(spark=spark, path=raw_csv_path)

    logger.info("Cleaning dataset...")
    df = sc.clean(
        raw_df=raw_df,
        drop_neutral_samples=drop_neutral_samples,
        text_col=sc.TEXT_COL,
        label_col=sc.LABEL_COL
    )

    # TODO ( ͡° ͜ʖ ͡°)
    raise NotImplementedError('Might need to do something here with custom features')

    df = ppb.preprocess_dataset(
        raw_df=df,
        drop_neutral_samples=drop_neutral_samples,
        text_col=sc.TEXT_COL,
        label_col=sc.LABEL_COL
    )

    dataset_path = WITH_NEUTRALS_DATASET_PATH if drop_neutral_samples else WITHOUT_NEUTRALS_DATASET_PATH
    logger.info("Preprocessing dataset...")
    df.write.parquet(str(dataset_path), mode='overwrite')
    logger.info("Preprocessing finished")


if __name__ == "__main__":
    _main()
