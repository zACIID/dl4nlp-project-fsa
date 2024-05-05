import typing

import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from transformers import AutoTokenizer, BatchEncoding

import data.spark as S
import data.stocktwits_crypto_dataset as sc
import data.common as common
import fine_tuned_finbert.models.fine_tuned_finbert as ft

_TOKENIZER_PATH = ft.PRE_TRAINED_MODEL_PATH

TEXT_COL = sc.TEXT_COL
LABEL_COL = common.LABEL_COL
TOKENIZER_OUTPUT_COL = "tokenizer"

PROCESSED_DATASET_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
    .add(TOKENIZER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
)


def preprocess_dataset(
        raw_df: psql.DataFrame,
        text_col: str,
) -> psql.DataFrame:
    """
    :param raw_df: just read, no preprocessing, raw dataset
    :param text_col: name of column in raw_df
    :return:
    """

    # Make sure the number of partitions is correct
    logger.info("Preprocessing corpus...")
    logger.debug(f"Number of RDD partitions: {raw_df.rdd.getNumPartitions()}")
    if raw_df.rdd.getNumPartitions() != S.EXECUTORS_AVAILABLE_CORES:
        logger.debug(f"Repartitioning RDD to {S.EXECUTORS_AVAILABLE_CORES}")
        raw_df = raw_df.repartition(numPartitions=S.EXECUTORS_AVAILABLE_CORES)

    logger.debug("Applying tokenizer...")
    with_tokens = _apply_tokenizer(df=raw_df, text_col=text_col)

    logger.debug("Preprocessing implemented")
    return with_tokens


def _apply_tokenizer(
        df: psql.DataFrame,
        text_col: str
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

    with_tokens_df = df.withColumn(TOKENIZER_OUTPUT_COL, tokenize(psqlf.col(text_col)))

    return with_tokens_df
