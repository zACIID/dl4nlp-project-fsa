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
import hand_eng_mlp_TODO.datasets.preprocessing_features_extraction as ppfe
import utils.io as io_
from hand_eng_mlp_TODO.datasets.custom_features import CustomFeatures
import hand_eng_mlp_TODO.models.model_beijin as hemlp


TEXT_COL = sc.TEXT_COL  # TODO this is actually different for each dataset
LABEL_COL = common.LABEL_COL
TOKENIZER_OUTPUT_COL = "tokenizer"
SENTIMENT_SCORE_COL = "sentiment_score"
_TOKENIZER_PATH = hemlp.PRE_TRAINED_MODEL_PATH

PROCESSED_DATASET_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
    .add(TOKENIZER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
    .add(SENTIMENT_SCORE_COL, psqlt.FloatType(), nullable=False)
)
# up to here to comment in case of failure


# Define UDFs for feature extraction functions
compute_sentence_polarity_VADER_udf = udf(
    ppfe.compute_vader_polarity, FloatType()
)
calculate_pos_neg_features_VADER_udf = udf(
    ppfe.calculate_vader_pos_neg_features,
    StructType([
        StructField("pos_neg_ratio_vader", FloatType()),
        StructField("pos_neg_difference_vader", FloatType())
    ])
)
calculate_sentiment_entropy_VADER_udf = udf(
    ppfe.calculate_vader_sentiment_entropy, FloatType()
)
calculate_pos_neg_features_SN_udf = udf(
    ppfe.calculate_sentic_pos_neg_features,
    StructType([
        StructField("pos_neg_ratio_sentic", FloatType()),
        StructField("pos_neg_difference_sentic", FloatType())
    ])
)
compute_sentence_polarity_SWN_udf = udf(
    ppfe.compute_swn_polarity, FloatType()
)
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
def get_new_features(df: psql.DataFrame, text_col: str) -> psql.DataFrame:
    """
    :param df: Spark DataFrame with text data
    :param text_col: name of the column containing the text
    :return: DataFrame with additional computed features
    """
    # # Sentiment score with VADER, SenticNet and SentiWordNet
    # df = df.withColumn('vader_polarity', compute_sentence_polarity_VADER_udf(df[text_col]))
    # print("END step1 - SAFE")

    # df = df.withColumn("vader_features", calculate_pos_neg_features_VADER_udf(df[text_col]))
    # df = df.select(
    #     "*",
    #     df["vader_features"]["pos_neg_ratio_vader"].alias("pos_neg_ratio_vader"),
    #     df["vader_features"]["pos_neg_difference_vader"].alias("pos_neg_difference_vader")
    # ).drop("vader_features")
    # print("END step2 - SAFE")

    # df = df.withColumn('sentiment_entropy_vader', calculate_sentiment_entropy_VADER_udf(df[text_col]))
    # print("END step3 - SAFE")



    # df = df.withColumn("senticnet_features", calculate_pos_neg_features_SN_udf(df[text_col]))
    # df = df.select(
    #     "*",
    #     df["senticnet_features"]["pos_neg_ratio_sentic"].alias("pos_neg_ratio_sentic"),
    #     df["senticnet_features"]["pos_neg_difference_sentic"].alias("pos_neg_difference_sentic")
    # ).drop("senticnet_features")
    # print("END step4 - SAFE?")  # TODO takes too much time, to run when sleeping, if the seconnd senticnet works this should works as well



    # df = df.withColumn('swn_polarity', compute_sentence_polarity_SWN_udf(df[text_col]))
    # print("END step5 - SAFE")



    # # Emotion recognition
    # df = df.withColumn("emotion_recognition", emotion_recognition_SN_udf(df[text_col]))
    # df = df.select(
    #     "*",
    #     df["emotion_recognition"]["INTROSPECTION"].alias("INTROSPECTION"),
    #     df["emotion_recognition"]["TEMPER"].alias("TEMPER"),
    #     df["emotion_recognition"]["ATTITUDE"].alias("ATTITUDE"),
    #     df["emotion_recognition"]["SENSITIVITY"].alias("SENSITIVITY")
    # ).drop("emotion_recognition")
    # print("END step6 - SAFE?")  # TODO takes too much time, to run when sleeping, if the seconnd senticnet works this should works as well



    # Readability metrics
    df = df.withColumn("readability_metrics", calculate_readability_metrics_udf(df[text_col]))
    df = df.select(
        "*",
        df["readability_metrics"]["flesch_kincaid_grade"].alias("flesch_kincaid_grade"),
        df["readability_metrics"]["gunning_fog"].alias("gunning_fog"),
        df["readability_metrics"]["coleman_liau_index"].alias("coleman_liau_index")
    ).drop("readability_metrics")
    print("END step7 - SAFE?")

    # # Lexical Affect Features: Valence, Arousal, Dominance (VAD)
    # # print("START load VAD dataset")  #TODO: remove later, ask ruie
    # # sentiment_data, mean_medians = ppfe.load_sentiment_dataset(io_.DATA_DIR)
    # # print("END load VAD dataset")
    # # df = df.withColumn("lexical_affect_features", compute_overall_sentiment_features_udf(df[text_col], sentiment_data, mean_medians))
    # df = df.withColumn("lexical_affect_features", compute_overall_sentiment_features_udf(df[text_col]))
    # df = df.select(
    #     "*",
    #     df["lexical_affect_features"]["overall_valence_mean"].alias("overall_valence_mean"),
    #     df["lexical_affect_features"]["overall_arousal_mean"].alias("overall_arousal_mean"),
    #     df["lexical_affect_features"]["overall_dominance_mean"].alias("overall_dominance_mean"),
    #     df["lexical_affect_features"]["overall_valence_std"].alias("overall_valence_std"),
    #     df["lexical_affect_features"]["overall_arousal_std"].alias("overall_arousal_std"),
    #     df["lexical_affect_features"]["overall_dominance_std"].alias("overall_dominance_std"),
    #     df["lexical_affect_features"]["valence_contrast"].alias("valence_contrast"),
    #     df["lexical_affect_features"]["arousal_contrast"].alias("arousal_contrast"),
    #     df["lexical_affect_features"]["dominance_contrast"].alias("dominance_contrast")
    # ).drop("lexical_affect_features")
    # print("END step8 - SAFE?")

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

    logger.debug("Applying tokenizer...")
    with_tokens = _apply_tokenizer(df=raw_df, text_col=text_col)

    logger.debug("Converting labels into sentiment scores (Bearish: -1, Neutral: 0, Bullish: 1)...")
    df = sc.convert_labels_to_sentiment_scores(df=with_tokens, label_col=label_col)

    # Extract additional features
    df = get_new_features(df, text_col=text_col)  #TODO fa errore wall of text numeri

    logger.debug("Preprocessing implemented")
    return df


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
