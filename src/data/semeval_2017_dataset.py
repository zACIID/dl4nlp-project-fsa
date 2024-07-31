import os
import typing
import json
import subprocess
import pyspark.sql as psql
from pathlib import Path
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf

import utils.io as io_


_DOWNLOAD_DIR = io_.RAW_DATASET_DIR / 'semeval2017'
_DOWNLOAD_BASH_COMMAND = f"git clone git@bitbucket.org:ssix-project/semeval-2017-task-5-subtask-1.git {_DOWNLOAD_DIR}"

# This is the maximum number of characters of the texts in the final test dataset (SemEval)
# Assuming one token per character, we have a maximum of 128 tokens,
#   meaning that I'd throw away the remaining characters/tokens
#   so that memory and training times do not explode
# TODO verify this statement -> we simply tried with the tokenizer and saw that no sentence was longer than 128
#  (the actual max is even less) -> this is also the max length of Bertweet
WORST_CASE_TOKENS = 128

SOURCE_COL = "source"
CASHTAG_COL = "cashtag"
SENTIMENT_SCORE_COL = "sentiment score"
ID_COL = "id"
TEXT_COL = "spans"

RAW_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.ArrayType(psqlt.StringType()), nullable=False)
    .add(SENTIMENT_SCORE_COL, psqlt.StringType(), nullable=False)
    .add(SOURCE_COL, psqlt.StringType(), nullable=False)
    .add(ID_COL, psqlt.StringType(), nullable=False)
    .add(CASHTAG_COL, psqlt.StringType(), nullable=False)
)
"""
JSON dataset schema, without *any* form of preprocessing. 
Schema may very well very after preprocessing.
"""


def download_dataset() -> Path:
    train_dataset_path = _DOWNLOAD_DIR / 'Microblog_Trainingdata.json'
    test_dataset_path = _DOWNLOAD_DIR / 'Microblog_Trialdata.json'
    combined_dataset_path = _DOWNLOAD_DIR / 'Combined_Dataset.json'

    # List of duplicate ids to remove from the training set
    duplicate_ids = {"708668814427348992", "34147106", "18479024", "16142438", "10752226"}

    if not os.path.exists(train_dataset_path):
        logger.info('Downloading training dataset...')
        subprocess.Popen(_DOWNLOAD_BASH_COMMAND,
                         stdout=subprocess.PIPE,
                         shell=True,
                         executable="/bin/bash"
                         ).communicate()
    else:
        logger.info('Training dataset already downloaded')

    if not os.path.exists(test_dataset_path):
        logger.info('Downloading test dataset...')
        subprocess.Popen(_DOWNLOAD_BASH_COMMAND,
                         stdout=subprocess.PIPE,
                         shell=True,
                         executable="/bin/bash"
                         ).communicate()
    else:
        logger.info('Test dataset already downloaded')

    with open(train_dataset_path, 'r', encoding='utf-8') as train_file:
        train_data = json.load(train_file)
    with open(test_dataset_path, 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)

    # Remove duplicates from training data and combine with the test data
    filtered_train_data = [sample for sample in train_data if sample['id'] not in duplicate_ids]
    combined_data = filtered_train_data + test_data

    with open(combined_dataset_path, 'w', encoding='utf-8') as combined_file:
        json.dump(combined_data, combined_file, ensure_ascii=False, indent=4)

    logger.info('Combined dataset saved')
    return combined_dataset_path


def read_dataset(
        spark: psql.SparkSession,
        path: Path,
) -> psql.DataFrame:
    df = spark.read.option("multiline", True).json(str(path))

    return df


def clean_dataset(df: psql.DataFrame) -> psql.DataFrame:
    # sentiment score column is string but we want float
    df = df.withColumn(SENTIMENT_SCORE_COL, psqlf.col(SENTIMENT_SCORE_COL).cast(psqlt.FloatType()))

    # Merge all spans into one text string
    @psqlf.udf(returnType=psqlt.StringType())
    def join_spans(spans: typing.List[str]) -> str:
        return " ".join(spans)

    df = df.withColumn(TEXT_COL, join_spans(psqlf.col(TEXT_COL)))
    df = df.withColumn(TEXT_COL, psqlf.when(psqlf.col(TEXT_COL) == "" ,None).otherwise(psqlf.col(TEXT_COL)))
    df = df.dropna(subset=[TEXT_COL, SENTIMENT_SCORE_COL])

    # Drop useless cols
    df = df.drop(SOURCE_COL, CASHTAG_COL)

    return df
