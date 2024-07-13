import typing
import pandas as pd
import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from pyspark.sql.functions import udf, col, struct, lit
from pyspark.sql.types import FloatType, StructType, StructField
from transformers import AutoTokenizer, BatchEncoding

import data.spark as S
import data.stocktwits_crypto_dataset as sc
import data.common as common
import fine_tuned_finbert.models.fine_tuned_finbert as ft
import hand_eng_mlp_TODO.datasets.preprocessing_features_extraction as ppfe
import utils.io as io_
from custom_features import CustomFeatures


# TODO ( ͡° ͜ʖ ͡°) compile once decided (in trial)
TEXT_COL = sc.TEXT_COL
LABEL_COL = common.LABEL_COL
TOKENIZER_OUTPUT_COL = "tokenizer"
SENTIMENT_SCORE_COL = "sentiment_score"

PROCESSED_DATASET_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
    .add(TOKENIZER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
    .add(SENTIMENT_SCORE_COL, psqlt.FloatType(), nullable=False)
)
# up to here to comment in case of failure


# Define UDFs for feature extraction functions
compute_sentence_polarity_VADER_udf = udf(ppfe.compute_vader_polarity, FloatType())
calculate_pos_neg_features_VADER_udf = udf(
    ppfe.calculate_vader_pos_neg_features,
    StructType([
        StructField("Pos_Neg_Ratio_VADER", FloatType()),
        StructField("Pos_Neg_Difference_VADER", FloatType())
    ])
)
calculate_sentiment_entropy_VADER_udf = udf(ppfe.calculate_vader_sentiment_entropy, FloatType())
calculate_pos_neg_features_SN_udf = udf(
    ppfe.calculate_sentic_pos_neg_features,
    StructType([
        StructField("Pos_Neg_Ratio_SenticNet", FloatType()),
        StructField("Pos_Neg_Difference_SenticNet", FloatType())
    ])
)
compute_sentence_polarity_SWN_udf = udf(ppfe.compute_swn_polarity, FloatType())
emotion_recognition_SN_udf = udf(
    ppfe.sentic_emotion_recognition,
    StructType([
        StructField("INTROSPECTION", FloatType()),
        StructField("TEMPER", FloatType()),
        StructField("ATTITUDE", FloatType()),
        StructField("SENSITIVITY", FloatType())
    ])
)
calculate_readability_metrics_udf = udf(
    ppfe.calculate_readability_metrics,
    StructType([
        StructField("flesch_kincaid_grade", FloatType()),
        StructField("gunning_fog", FloatType()),
        StructField("coleman_liau_index", FloatType())
    ])
)
compute_overall_sentiment_features_udf = udf(
    ppfe.compute_overall_sentiment_features,
    StructType([
        StructField("overall_valence_mean", FloatType()),
        StructField("overall_arousal_mean", FloatType()),
        StructField("overall_dominance_mean", FloatType()),
        StructField("overall_valence_std", FloatType()),
        StructField("overall_arousal_std", FloatType()),
        StructField("overall_dominance_std", FloatType()),
        StructField("valence_contrast", FloatType()),
        StructField("arousal_contrast", FloatType()),
        StructField("dominance_contrast", FloatType())
    ])
)


# Function to compute the additional features
def get_new_features(df: psql.DataFrame) -> psql.DataFrame:
    """
    :param df: Spark DataFrame with text data
    :return: DataFrame with additional computed features
    """
    features = CustomFeatures()

    # Sentiment score with VADER, SenticNet and SentiWordNet
    df = df.withColumn('sentiment_score_VADER', compute_sentence_polarity_VADER_udf(df['text']))

    pos_neg_vader_udf = calculate_pos_neg_features_VADER_udf(df['text'])
    df = df.withColumn('Pos_Neg_Ratio_VADER', pos_neg_vader_udf['Pos_Neg_Ratio_VADER'])
    df = df.withColumn('Pos_Neg_Difference_VADER', pos_neg_vader_udf['Pos_Neg_Difference_VADER'])

    df = df.withColumn('Sentiment_Entropy_VADER', calculate_sentiment_entropy_VADER_udf(df['text']))

    pos_neg_sn_udf = calculate_pos_neg_features_SN_udf(df['text'])
    df = df.withColumn('Pos_Neg_Ratio_SenticNet', pos_neg_sn_udf['Pos_Neg_Ratio_SenticNet'])
    df = df.withColumn('Pos_Neg_Difference_SenticNet', pos_neg_sn_udf['Pos_Neg_Difference_SenticNet'])

    df = df.withColumn('sentiment_score_SWN', compute_sentence_polarity_SWN_udf(df['text']))

    # Emotion recognition
    emotion_features_udf = emotion_recognition_SN_udf(df['text'])
    df = df.withColumn('INTROSPECTION', emotion_features_udf['INTROSPECTION'])
    df = df.withColumn('TEMPER', emotion_features_udf['TEMPER'])
    df = df.withColumn('ATTITUDE', emotion_features_udf['ATTITUDE'])
    df = df.withColumn('SENSITIVITY', emotion_features_udf['SENSITIVITY'])

    # Readability metrics
    readability_metrics_udf = calculate_readability_metrics_udf(df['text'])
    df = df.withColumn('flesch_kincaid_grade', readability_metrics_udf['flesch_kincaid_grade'])
    df = df.withColumn('gunning_fog', readability_metrics_udf['gunning_fog'])
    df = df.withColumn('coleman_liau_index', readability_metrics_udf['coleman_liau_index'])

    # Lexical Affect Features: Valence, Arousal, Dominance (VAD)
    sentiment_data, mean_medians = ppfe.load_sentiment_dataset(io_.DATA_DIR)
    oa_sent_features_udf = compute_overall_sentiment_features_udf(df['text'], lit(sentiment_data), lit(mean_medians))
    df = df.withColumn('overall_valence_mean', oa_sent_features_udf['overall_valence_mean'])
    df = df.withColumn('overall_arousal_mean', oa_sent_features_udf['overall_arousal_mean'])
    df = df.withColumn('overall_dominance_mean', oa_sent_features_udf['overall_dominance_mean'])
    df = df.withColumn('overall_valence_std', oa_sent_features_udf['overall_valence_std'])
    df = df.withColumn('overall_arousal_std', oa_sent_features_udf['overall_arousal_std'])
    df = df.withColumn('overall_dominance_std', oa_sent_features_udf['overall_dominance_std'])
    df = df.withColumn('valence_contrast', oa_sent_features_udf['valence_contrast'])
    df = df.withColumn('arousal_contrast', oa_sent_features_udf['arousal_contrast'])
    df = df.withColumn('dominance_contrast', oa_sent_features_udf['dominance_contrast'])

    return df


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

    # Extract additional features
    df = get_new_features(df)  #TODO tokenize first before adding the custom features(?)

    logger.debug("Preprocessing implemented")
    return df
