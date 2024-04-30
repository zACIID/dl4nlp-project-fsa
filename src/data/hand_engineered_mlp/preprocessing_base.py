import typing

import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from transformers import AutoTokenizer, BatchEncoding

import data.spark as S
import data.stocktwits_crypto_dataset as sc
import data.common as common
import models.fine_tuned_finbert as ft


# TODO compile once decided
# TEXT_COL = sc.TEXT_COL
# LABEL_COL = common.LABEL_COL
# TOKENIZER_OUTPUT_COL = "tokenizer"
# SENTIMENT_SCORE_COL = "sentiment_score"
#
# PROCESSED_DATASET_SCHEMA: psqlt.StructType = (
#     psqlt.StructType()
#     .add(TEXT_COL, psqlt.StringType(), nullable=False)
#     .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
#     .add(TOKENIZER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
#     .add(SENTIMENT_SCORE_COL, psqlt.FloatType(), nullable=False)
# )
#

def preprocess_dataset(
        raw_df: psql.DataFrame,
        drop_neutral_samples: bool,
        text_col: str,
        label_col: str
) -> psql.DataFrame:
    """
    :param raw_df: just read, no preprocessing, raw dataset
    :param drop_neutral_samples: true if neutrally labelled samples should be dropped
    :param label_col: name of column in raw_df
    :param text_col: name of column in raw_df
    :return:
    """

    # Make sure the number of partitions is correct
    logger.info("Preprocessing corpus...")
    logger.debug(f"Number of RDD partitions: {raw_df.rdd.getNumPartitions()}")
    if raw_df.rdd.getNumPartitions() != S.EXECUTORS_AVAILABLE_CORES:
        logger.debug(f"Repartitioning RDD to {S.EXECUTORS_AVAILABLE_CORES}")
        raw_df = raw_df.repartition(numPartitions=S.EXECUTORS_AVAILABLE_CORES)

    logger.debug("Converting labels into sentiment scores (Bearish: -1, Neutral: 0, Bullish: 1)...")
    df = sc.convert_labels_to_sentiment_scores(df=raw_df, label_col=label_col)

    # TODO do something here with the text: craft the custom features
    raise NotImplementedError('TODO')

    logger.debug("Preprocessing implemented")
    return with_scores_df
