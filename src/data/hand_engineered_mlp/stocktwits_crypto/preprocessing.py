import os
import typing
import urllib.request
from pathlib import Path

import click
import datasets
import pandas as pd
import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from transformers import AutoTokenizer, BatchEncoding

import data.spark as S
import data.stocktwits_crypto_dataset as sc
import utils.io as io_


_MODEL_NAME = 'hand_eng_mlp'
_SPARK_APP_NAME = f'{_MODEL_NAME} Preprocessing'

# # TODO example of output dataset schema
# DATASET_SCHEMA: psqlt.StructType = (
#     psqlt.StructType()
#     .add(sc.TEXT_COL, psqlt.StringType(), nullable=False)
#     .add(sc.LABEL_COL, psqlt.IntegerType(), nullable=False)
#     .add(TOKENIZER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
#     .add(SENTIMENT_SCORE_COL, psqlt.FloatType(), nullable=False)
# )

WITH_NEUTRALS_DATASET_PATH = io_.DATA_DIR / f'stocktwits-crypto-{_MODEL_NAME}-with-neutrals.parquet'
WITHOUT_NEUTRALS_DATASET_PATH = io_.DATA_DIR / f'stocktwits-crypto-{_MODEL_NAME}-without-neutrals.parquet'

SENTIMENT_SCORE_COL = "sentiment_score"


def get_dataset(drop_neutral_samples: bool) -> datasets.Dataset:
    dataset_path = WITH_NEUTRALS_DATASET_PATH if drop_neutral_samples else WITHOUT_NEUTRALS_DATASET_PATH
    if not os.path.exists(dataset_path):
        raise FileNotFoundError('Dataset not found. Make sure to run this script to execute '
                                'the preprocessing pipeline for this dataset')

    # Have to load the actual .parquet files inside the dataset folder
    return datasets.Dataset.from_parquet(str(dataset_path / "*.parquet"))


def _get_document_features(
        spark: psql.SparkSession,
        corpus_csv_path: Path,
        corpus_schema: psqlt.StructType,
        drop_neutral_samples: bool
) -> psql.DataFrame:
    logger.info("Loading corpus...")
    raw_df: psql.DataFrame = _read_corpus(
        spark=spark,
        corpus_csv_path=corpus_csv_path,
        corpus_schema=corpus_schema
    )

    # Make sure the number of partitions is correct
    logger.info("Preprocessing corpus...")
    logger.debug(f"Number of RDD partitions: {raw_df.rdd.getNumPartitions()}")
    if raw_df.rdd.getNumPartitions() != S.EXECUTORS_AVAILABLE_CORES:
        logger.debug(f"Repartitioning RDD to {S.EXECUTORS_AVAILABLE_CORES}")
        raw_df = raw_df.repartition(numPartitions=S.EXECUTORS_AVAILABLE_CORES)

    logger.info("Cleaning data...")
    df = _clean(raw_df=raw_df, drop_neutral_samples=drop_neutral_samples)

    # TODO do some stuff here

    logger.debug("Converting labels into sentiment scores (Bearish: -1, Neutral: 0, Bullish: 1)...")
    df = _convert_labels_to_sentiment_scores(df=df)

    logger.debug("Preprocessing implemented")
    return df


def _read_corpus(
        spark: psql.SparkSession,
        corpus_csv_path: Path,
        corpus_schema: psqlt.StructType
) -> psql.DataFrame:
    raw_docs_df: psql.DataFrame = spark.read.csv(str(corpus_csv_path), header=True, schema=corpus_schema)

    return raw_docs_df


def _clean(raw_df: psql.DataFrame, drop_neutral_samples: bool) -> psql.DataFrame:
    raw_df = raw_df.fillna({sc.TEXT_COL: "", sc.LABEL_COL: 1})  # 1 is neutral label in raw dataset

    if drop_neutral_samples:
        # Drop neutral labels because they add noise:
        #   the absence of label is what defines them as neutral,
        #   meaning that they could in actuality express positive or negative.
        # Neutral label may hence prove misleading
        raw_df = raw_df.filter(f"{sc.LABEL_COL} <> 1")

    return raw_df


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

    df = _get_document_features(
        spark=spark,
        corpus_csv_path=raw_csv_path,
        corpus_schema=sc.SCHEMA,
        drop_neutral_samples=drop_neutral_samples
    )

    dataset_path = WITH_NEUTRALS_DATASET_PATH if drop_neutral_samples else WITHOUT_NEUTRALS_DATASET_PATH
    logger.info("Preprocessing dataset...")
    df.write.parquet(str(dataset_path), mode='overwrite')
    logger.info("Preprocessing finished")


if __name__ == "__main__":
    _main()
