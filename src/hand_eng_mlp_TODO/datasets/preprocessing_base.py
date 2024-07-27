import pyspark.sql as psql
import torch
from loguru import logger
from pyspark.sql import types as psqlt
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StructType, StructField, ArrayType
from transformers import AutoTokenizer, AutoModel

import data.common as common
import data.spark as S
import data.stocktwits_crypto_dataset as sc
import hand_eng_mlp_TODO.datasets.preprocessing_features_extraction as ppfe
import hand_eng_mlp_TODO.models.model_beijin as hemlp
import utils.io as io_

TEXT_COL = sc.TEXT_COL  # TODO this is actually different for each dataset
LABEL_COL = common.LABEL_COL
EMBEDDER_OUTPUT_COL = "embedder"
SENTIMENT_SCORE_COL = "sentiment_score"
_TOKENIZER_PATH = hemlp.PRE_TRAINED_MODEL_PATH

PROCESSED_DATASET_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
    .add(EMBEDDER_OUTPUT_COL, psqlt.ArrayType(psqlt.IntegerType()), nullable=False)
    .add(SENTIMENT_SCORE_COL, psqlt.FloatType(), nullable=False)
)

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


def create_compute_overall_sentiment_features_udf(sentiment_data_broadcast, default_mean_value_broadcast):
    sentiment_data = sentiment_data_broadcast.value
    default_mean_value = default_mean_value_broadcast.value

    def compute_overall_sentiment_features_closure(text: str) -> ppfe.cf.SentimentFeatures:
        return ppfe.compute_overall_sentiment_features(
            text,
            sentiment_data,
            default_mean_value
        )

    return udf(
        compute_overall_sentiment_features_closure,
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
def get_new_features(
        spark: psql.SparkSession,
        df: psql.DataFrame,
        text_col: str) -> psql.DataFrame:
    """
    :param spark: spark session
    :param df: Spark DataFrame with text data
    :param text_col: name of the column containing the text
    :return: DataFrame with additional computed features
    """
    # Sentiment score with VADER and SentiWordNet
    df = df.withColumn('vader_polarity', compute_sentence_polarity_VADER_udf(df[text_col]))

    df = df.withColumn("vader_features", calculate_pos_neg_features_VADER_udf(df[text_col]))
    df = df.select(
        "*",
        df["vader_features"]["pos_neg_ratio_vader"].alias("pos_neg_ratio_vader"),
        df["vader_features"]["pos_neg_difference_vader"].alias("pos_neg_difference_vader")
    ).drop("vader_features")

    df = df.withColumn('sentiment_entropy_vader', calculate_sentiment_entropy_VADER_udf(df[text_col]))

    df = df.withColumn('swn_polarity', compute_sentence_polarity_SWN_udf(df[text_col]))

    # Emotion recognition with SenticNet
    df = df.withColumn("emotion_recognition", emotion_recognition_SN_udf(df[text_col]))
    df = df.select(
        "*",
        df["emotion_recognition"]["INTROSPECTION"].alias("INTROSPECTION"),
        df["emotion_recognition"]["TEMPER"].alias("TEMPER"),
        df["emotion_recognition"]["ATTITUDE"].alias("ATTITUDE"),
        df["emotion_recognition"]["SENSITIVITY"].alias("SENSITIVITY")
    ).drop("emotion_recognition")

    # Readability metrics
    df = df.withColumn("readability_metrics", calculate_readability_metrics_udf(df[text_col]))
    df = df.select(
        "*",
        df["readability_metrics"]["flesch_kincaid_grade"].alias("flesch_kincaid_grade"),
        df["readability_metrics"]["gunning_fog"].alias("gunning_fog"),
        df["readability_metrics"]["coleman_liau_index"].alias("coleman_liau_index")
    ).drop("readability_metrics")

    # Lexical Affect Features: Valence, Arousal, Dominance (VAD)
    sentiment_data, default_mean_value = ppfe.load_sentiment_dataset(io_.DATA_DIR)
    sentiment_data_broadcast = spark.sparkContext.broadcast(sentiment_data)
    default_mean_value_broadcast = spark.sparkContext.broadcast(default_mean_value)

    compute_overall_sentiment_features_udf = create_compute_overall_sentiment_features_udf(
        sentiment_data_broadcast,
        default_mean_value_broadcast
    )

    df = df.withColumn("lexical_affect_features", compute_overall_sentiment_features_udf(df[text_col]))
    df = df.select(
        "*",
        df["lexical_affect_features"]["overall_valence_mean"].alias("overall_valence_mean"),
        df["lexical_affect_features"]["overall_arousal_mean"].alias("overall_arousal_mean"),
        df["lexical_affect_features"]["overall_dominance_mean"].alias("overall_dominance_mean"),
        df["lexical_affect_features"]["overall_valence_std"].alias("overall_valence_std"),
        df["lexical_affect_features"]["overall_arousal_std"].alias("overall_arousal_std"),
        df["lexical_affect_features"]["overall_dominance_std"].alias("overall_dominance_std"),
        df["lexical_affect_features"]["valence_contrast"].alias("valence_contrast"),
        df["lexical_affect_features"]["arousal_contrast"].alias("arousal_contrast"),
        df["lexical_affect_features"]["dominance_contrast"].alias("dominance_contrast")
    ).drop("lexical_affect_features")

    return df


def preprocess_dataset(
        spark: psql.SparkSession,
        raw_df: psql.DataFrame,
        drop_neutral_samples: bool,
        text_col: str,
        label_col: str
) -> psql.DataFrame:
    """
    :param spark: spark session
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
    with_embeds = _apply_tokenize_and_embed(df=raw_df, text_col=text_col)

    logger.debug("Converting labels into sentiment scores (Bearish: -1, Neutral: 0, Bullish: 1)...")
    df = sc.convert_labels_to_sentiment_scores(df=with_embeds, label_col=label_col)

    # Extract additional features
    df = get_new_features(spark, df, text_col=text_col)

    logger.debug("Preprocessing implemented")
    return df


def _apply_tokenize_and_embed( # TODO problem of too much data?
        df: psql.DataFrame,
        text_col: str
) -> psql.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(hemlp.PRE_TRAINED_MODEL_PATH, use_fast=True)
    bertweet = AutoModel.from_pretrained(hemlp.PRE_TRAINED_MODEL_PATH)
    bertweet.eval()

    def tokenize_and_embed(text: str) -> list:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = bertweet(**inputs)  # **inputs unpacks the dictionary returned by the tokenizer

        # Exclude first (CLS) and last (SEP) tokens
        embeddings = outputs.last_hidden_state[:, 1:-1, :].numpy()

        return [embedding.tolist() for embedding in embeddings]

    tokenize_and_embed_udf = udf(tokenize_and_embed, ArrayType(ArrayType(FloatType())))
    df_with_embeddings = df.withColumn('embeddings', tokenize_and_embed_udf(df[text_col]))
    return df_with_embeddings


def _apply_embedder_temporarily_disabled(  # TODO old code, gives error
        df: psql.DataFrame,
        text_col: str
) -> psql.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(hemlp.PRE_TRAINED_MODEL_PATH, use_fast=True)
    bertweet = AutoModel.from_pretrained(hemlp.PRE_TRAINED_MODEL_PATH)

    @psqlf.udf(
        returnType=psqlt.StructType([ #todo: what should be the returning type og the udf?
            psqlt.StructField("input_ids", psqlt.ArrayType(psqlt.IntegerType())),
            psqlt.StructField("attention_mask", psqlt.ArrayType(psqlt.IntegerType()))
        ])
    )
    def tokenize(texts) -> typing.List:
        # NOTE: UDFs complex types are defined as StructType
        # - https://stackoverflow.com/a/53346512
        # - https://stackoverflow.com/a/36841721
        # batch: BatchEncoding = tokenizer(
        #     text if text is not None else "",
        #     return_tensors='np',
        #     return_attention_mask=True,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=sc.WORST_CASE_TOKENS
        # )
        #
        # return torch.tensor([tokenizer.encode(batch)])

        for idx, text in enumerate(texts):
            texts[idx] = tokenizer.encode(text)

        return texts
        # TODO check type:
        #  as we can see from colab file, torch.tensor([tokenizer.encode(line)]) returns a tensor([[...], [...], ...])
        #  is the type correct? should we not convert to tensor now but do it later below?
        #  pyspark might not like torch tensor type output
        #  try sparkdf.apply(column, function)

    with torch.no_grad():
        print("Tipo del testo: ", type(psqlf.col(text_col)))
        batch = tokenize(psqlf.col(text_col))
        print(type(batch))
        features = bertweet(torch.tensor(batch))
        print(type(features))

    embeds = features.hidden_states[-1]
    embeds = embeds[:, 1:-1, :]  # we don't need first and last embeddings because they are CLS and SEP tokens

    with_embeds_df = df.withColumn(EMBEDDER_OUTPUT_COL, list(embeds))  # TODO check if necessary to list()

    # TODO run preprocessing and see if everything works before the training TODOs

    return with_embeds_df
