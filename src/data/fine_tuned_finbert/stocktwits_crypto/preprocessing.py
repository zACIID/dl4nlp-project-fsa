import os
import typing
from pathlib import Path

import click
import datasets
import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from transformers import AutoTokenizer, BatchEncoding

import models.fine_tuned_finbert as ft
import data.spark as S
import data.stocktwits_crypto_dataset as sc
import utils.io as io_


_MODEL_NAME = 'finbert'
_SPARK_APP_NAME = f'{_MODEL_NAME} Preprocessing'

_TOKENIZER_PATH = ft.PRE_TRAINED_MODEL_PATH

TOKENIZER_OUTPUT_COL = "tokenizer"
SENTIMENT_SCORE_COL = "sentiment_score"

DATASET_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(sc.TEXT_COL, psqlt.StringType(), nullable=False)
    .add(sc.LABEL_COL, psqlt.IntegerType(), nullable=False)
    .add(TOKENIZER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
    .add(SENTIMENT_SCORE_COL, psqlt.FloatType(), nullable=False)
)

WITH_NEUTRALS_DATASET_PATH = io_.DATA_DIR / f'stocktwits-crypto-{_MODEL_NAME}-with-neutrals.parquet'
WITHOUT_NEUTRALS_DATASET_PATH = io_.DATA_DIR / f'stocktwits-crypto-{_MODEL_NAME}-without-neutrals.parquet'

#
# def get_dataset(drop_neutral_samples: bool) -> datasets.Dataset:
    # dataset_path = WITH_NEUTRALS_DATASET_PATH if drop_neutral_samples else WITHOUT_NEUTRALS_DATASET_PATH
    # if not os.path.exists(dataset_path):
    #     raise FileNotFoundError('Dataset not found. Make sure to run this script to execute '
    #                             'the preprocessing pipeline for this dataset')
    #
    # # Have to load the actual .parquet files inside the dataset folder
    # return datasets.Dataset.from_parquet(str(dataset_path / "*.parquet"))

# TODO change what is below last so interface doesn't brake while training still running

DATASET_PATH = io_.DATA_DIR / 'stocktwits-crypto.parquet' # TODO remove after uncommenting above

def get_dataset() -> datasets.Dataset:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError('Dataset not found. Make sure to run this script to execute '
                                'the preprocessing pipeline for this dataset')

    # Have to load the actual .parquet files inside the dataset folder
    return datasets.Dataset.from_parquet(str(DATASET_PATH / "*.parquet"))


# TODO this should be made generic: accept a dataset or a read_dataset callable or an enum that allows choice
def _preprocess_dataset(
        spark: psql.SparkSession,
        dataset_path: Path,
        drop_neutral_samples: bool
) -> psql.DataFrame:
    logger.info("Loading corpus...")
    raw_df: psql.DataFrame = sc.read_dataset(
        spark=spark,
        path=dataset_path,
    )

    # Make sure the number of partitions is correct
    logger.info("Preprocessing corpus...")
    logger.debug(f"Number of RDD partitions: {raw_df.rdd.getNumPartitions()}")
    if raw_df.rdd.getNumPartitions() != S.EXECUTORS_AVAILABLE_CORES:
        logger.debug(f"Repartitioning RDD to {S.EXECUTORS_AVAILABLE_CORES}")
        raw_df = raw_df.repartition(numPartitions=S.EXECUTORS_AVAILABLE_CORES)

    logger.info("Cleaning data...")
    with_na_filled = sc.clean(raw_df=raw_df, drop_neutral_samples=drop_neutral_samples)

    logger.debug("Applying tokenizer...")
    with_tokens = _apply_tokenizer(df=with_na_filled)

    logger.debug("Converting labels into sentiment scores (Bearish: -1, Neutral: 0, Bullish: 1)...")
    with_scores_df = _convert_labels_to_sentiment_scores(df=with_tokens)

    logger.debug("Preprocessing implemented")
    return with_scores_df


def _apply_tokenizer(
        df: psql.DataFrame
) -> psql.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, use_fast=True)

    @psqlf.udf(
        returnType=psqlt.StructType([
            psqlt.StructField("input_ids", psqlt.ArrayType(psqlt.IntegerType())),
            psqlt.StructField("attention_mask", psqlt.ArrayType(psqlt.IntegerType()))
        ])
    )
    def tokenize(text: str) -> typing.Tuple:
        # NOTE: UDFs complex types are defined as StructType
        # - https://stackoverflow.com/a/53346512
        # - https://stackoverflow.com/a/36841721
        batch: BatchEncoding = tokenizer(
            text if text is not None else "",
            return_tensors='np',
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            max_length=sc.WORST_CASE_TOKENS
        )
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()

        return input_ids.tolist(), attention_mask.tolist()

    with_tokens_df = df.withColumn(TOKENIZER_OUTPUT_COL, tokenize(psqlf.col(sc.TEXT_COL)))

    return with_tokens_df


def _convert_labels_to_sentiment_scores(
        df: psql.DataFrame
) -> psql.DataFrame:
    @psqlf.udf(returnType=psqlt.FloatType())
    def convert_label(label: int) -> float:
        match label:
            case 0.0: return -1.0
            case 1.0: return 0.0
            case 2.0: return 1.0
            case _: raise ValueError(f'Unknown label {label}')

    with_sent_score_df = df.withColumn(SENTIMENT_SCORE_COL, convert_label(psqlf.col(sc.LABEL_COL)))

    return with_sent_score_df


@click.command(
    help="Preprocess FinBERT fine-tuning dataset"
)
@click.option("--drop-neutral-samples", '-d', is_flag=True, type=click.BOOL)
def _main(drop_neutral_samples: bool):
    raw_csv_path = sc.download_dataset()

    spark = S.create_spark_session(
        app_name=_SPARK_APP_NAME,
    )

    df = _preprocess_dataset(
        spark=spark,
        dataset_path=raw_csv_path,
        drop_neutral_samples=drop_neutral_samples
    )

    dataset_path = WITH_NEUTRALS_DATASET_PATH if drop_neutral_samples else WITHOUT_NEUTRALS_DATASET_PATH
    logger.info("Preprocessing dataset...")
    df.write.parquet(str(dataset_path), mode='overwrite')
    logger.info("Preprocessing finished")


if __name__ == "__main__":
    _main()
