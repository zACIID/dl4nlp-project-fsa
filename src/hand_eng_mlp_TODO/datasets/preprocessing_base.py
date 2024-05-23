import typing

import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from transformers import AutoTokenizer, BatchEncoding

import data.spark as S
import data.stocktwits_crypto_dataset as sc
import data.common as common
import fine_tuned_finbert.models.fine_tuned_finbert as ft
import hand_eng_mlp.datasets_TODO.preprocessing_features_extraction as ppfe


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

    # TODO ( ͡° ͜ʖ ͡°) do something here with the text: craft the custom features
    # Apply the function to each row of the DataFrame
    df = df.rdd.map(lambda row: get_new_features(row)).toDF()
    # raise NotImplementedError('TODO')

    logger.debug("Preprocessing implemented")
    return df


# TODO ruie: adapt pandas dataframe to pyspark dataframe, and rrewrite lowercase functin
# Function to compute the additional features
def get_new_features(df):
    additional_features = []

    # Sentiment score with VADER, SenticNet and SentiWordNet]
    df['sentiment_score_VADER'] = df['text'].apply(ppfe.compute_sentence_polarity_VADER)
    df['Pos_Neg_Ratio_VADER'], df['Pos_Neg_Difference_VADER'] = zip(*df['text'].apply(ppfe.calculate_pos_neg_features_VADER))
    df['Sentiment_Entropy_VADER'] = df['text'].apply(ppfe.calculate_sentiment_entropy_VADER)
    df['Pos_Neg_Ratio_SenticNet'], df['Pos_Neg_Difference_SenticNet'] = zip(*df['text'].apply(ppfe.calculate_pos_neg_features_SN))
    df['sentiment_score_SWN'] = df['text'].apply(ppfe.compute_sentence_polarity_SWN)

    # Emotion recognition
    df['emotion_features'] = df['text'].apply(ppfe.emotion_recognition_SN)
    df[['INTROSPECTION', 'TEMPER', 'ATTITUDE', 'SENSITIVITY']] = df['emotion_features'].apply(pd.Series)
    df.drop(columns=['emotion_features'], inplace=True)

    # Readability metrics
    df['flesch_kincaid_grade'], df['gunning_fog'], df['coleman_liau_index'] = zip(*df['text'].apply(ppfe.calculate_readability_metrics))

    # Lexical Affect Features: Valence, Arousal, Dominance (VAD)
    sentiment_dataset_path = 'BRM-emot-submit.csv'
    sentiment_data, mean_medians = ppfe.load_sentiment_dataset(sentiment_dataset_path)
    df['overall_valence_mean'], df['overall_arousal_mean'], df['overall_dominance_mean'], df['overall_valence_std'], df['overall_arousal_std'], df['overall_dominance_std'], df['valence_contrast'], df['arousal_contrast'], df['dominance_contrast'] = zip(*df['text'].apply(ppfe.compute_overall_sentiment_features, sentiment_data=sentiment_data, default_mean_value=mean_medians))

    # Convert each element in the DataFrame to float if possible
    df = df.applymap(lambda x: float(x) if isinstance(x, (int, float)) else x)

    return df
