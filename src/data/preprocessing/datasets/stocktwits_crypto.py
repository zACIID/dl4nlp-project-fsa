import os
import typing
import urllib.request
from pathlib import Path

import datasets
import pandas as pd
import pyspark.sql as psql
import torch
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from transformers import AutoTokenizer, BatchEncoding

import data.preprocessing.spark as S
import utils.io as io_

_DOWNLOAD_URL = 'https://huggingface.co/datasets/ElKulako/stocktwits-crypto/resolve/main/st-data-full.xlsx?download=true'
_TOKENIZER_PATH = 'ahmedrachid/FinancialBERT-Sentiment-Analysis'  # TODO this should not be hardcoded, especially if we decide to use different pre-trained models

# This is the maximum number of characters of the texts in the final test dataset (SemEval)
# Assuming one token per character, we have a maximum of 160 tokens,
#   meaning that I'd throw away the remaining characters/tokens
#   so that memory and training times do not explode
WORST_CASE_TOKENS = 160

TEXT_COL = "text"
LABEL_COL = "label"

_RAW_CORPUS_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
)

TOKENIZER_OUTPUT_COL = "tokenizer"
SENTIMENT_SCORE_COL = "sentiment_score"

DATASET_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
    .add(TOKENIZER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
    .add(SENTIMENT_SCORE_COL, psqlt.FloatType(), nullable=False)
)

DATASET_PATH = io_.DATA_DIR / 'stocktwits-crypto.parquet'


def get_dataset() -> datasets.Dataset:
    # TODO maybe put this inside a class with a common interface? maybe not necessary because I will
    #   call this function from each training script so I won't use the interface effectively
    # TODO 2: this should preprocess the dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError('Dataset not found. Make sure to run this script to execute '
                                'the preprocessing pipeline for this dataset')

    # Have to load the actual .parquet files inside the dataset folder
    return datasets.Dataset.from_parquet(str(DATASET_PATH / "*.parquet"))


def _download_dataset(url: str) -> Path:
    raw_xlsx_path = io_.RAW_DATASET_DIR / 'stocktwits-crypto.xlsx'
    raw_csv_path = io_.RAW_DATASET_DIR / 'stocktwits-crypto.csv'

    if not os.path.exists(raw_xlsx_path) or not os.path.exists(raw_csv_path):
        logger.info('Downloading stocktwits-crypto dataset from {}'.format(url))
        with io_.DownloadProgressBar(
                unit='B',
                unit_scale=True,
                miniters=1,
                desc=url.split('/')[-1]
        ) as t:
            urllib.request.urlretrieve(url, filename=raw_xlsx_path, reporthook=t.update_to)

        logger.info('Converting from .xlsx to .csv')
        # There is also this Spark plugin to directly handle .xlsx in Spark,
        #   but I do not want to set it up and check how it works:
        #   https://github.com/crealytics/spark-excel
        df = pd.read_excel(raw_xlsx_path, sheet_name=[0, 1])  # there are two sheets in that xlsx file
        merged = pd.concat(df.values(), axis='rows')
        merged.to_csv(raw_csv_path, encoding='utf-8', index=False)
    else:
        logger.info('Dataset already downloaded')

    return raw_csv_path


def _get_document_features(
        spark: psql.SparkSession,
        corpus_csv_path: Path,
        corpus_schema: psqlt.StructType
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

    with_na_filled = _fillna(raw_df=raw_df)

    logger.debug("Applying tokenizer...")
    with_tokens = _apply_tokenizer(df=with_na_filled)

    logger.debug("Converting labels into sentiment scores (Bearish: -1, Neutral: 0, Bullish: 1)...")
    with_scores_df = _convert_labels_to_sentiment_scores(df=with_tokens)

    logger.debug("Corpus successfully preprocessed")
    return with_scores_df


def _read_corpus(
        spark: psql.SparkSession,
        corpus_csv_path: Path,
        corpus_schema: psqlt.StructType
) -> psql.DataFrame:
    raw_docs_df: psql.DataFrame = spark.read.csv(str(corpus_csv_path), header=True, schema=corpus_schema)

    return raw_docs_df


def _fillna(raw_df: psql.DataFrame) -> psql.DataFrame:
    return raw_df.fillna({TEXT_COL: "", LABEL_COL: 1})  # 1 is neutral label in raw dataset


def _apply_tokenizer(
        df: psql.DataFrame
) -> psql.DataFrame:
    # TODO I do not know if it is efficient to use tokenizers like this and if use_fast here raises the TOKENIZER_PARALLELISM problem
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, use_fast=True)

    @psqlf.udf(
        returnType=psqlt.StructType([
            psqlt.StructField("token_ids", psqlt.ArrayType(psqlt.IntegerType())),
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
            max_length=WORST_CASE_TOKENS
        )
        token_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        token_ids = token_ids.squeeze()
        attention_mask = attention_mask.squeeze()

        return token_ids.tolist(), attention_mask.tolist()

    with_tokens_df = df.withColumn(TOKENIZER_OUTPUT_COL, tokenize(psqlf.col(TEXT_COL)))

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

    with_sent_score_df = df.withColumn(SENTIMENT_SCORE_COL, convert_label(psqlf.col(LABEL_COL)))

    return with_sent_score_df


def _main():
    # TODO maybe when I'll have implemented all the other pre processing pipelines I'll have a better idea on how
    #   to make a better CLI and dataset interface for all the scripts
    raw_csv_path = _download_dataset(url=_DOWNLOAD_URL)

    spark = S.create_spark_session(
        app_name=f"[Dataset Preprocessing] {_DOWNLOAD_URL}",
    )

    df = _get_document_features(
        spark=spark,
        corpus_csv_path=raw_csv_path,
        corpus_schema=_RAW_CORPUS_SCHEMA
    )
    df.write.parquet(str(DATASET_PATH), mode='overwrite')


if __name__ == "__main__":
    _main()
