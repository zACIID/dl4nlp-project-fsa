import os
import typing
from pathlib import Path
import subprocess

import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf

import utils.io as io_


_DOWNLOAD_DIR = io_.RAW_DATASET_DIR / 'semeval2017'
_DOWNLOAD_BASH_COMMAND = f"git clone git@bitbucket.org:ssix-project/semeval-2017-task-5-subtask-1.git {_DOWNLOAD_DIR}"

# This is the maximum number of characters of the texts in the final test dataset (SemEval)
# Assuming one token per character, we have a maximum of 160 tokens,
#   meaning that I'd throw away the remaining characters/tokens
#   so that memory and training times do not explode
WORST_CASE_TOKENS = 160

SOURCE_COL = "source"
CASHTAG_COL = "cashtag"
LABEL_COL = "sentiment score"
ID_COL = "id"
TEXT_COL = "spans"

RAW_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.ArrayType(psqlt.StringType()), nullable=False)
    .add(LABEL_COL, psqlt.StringType(), nullable=False)
    .add(SOURCE_COL, psqlt.StringType(), nullable=False)
    .add(ID_COL, psqlt.StringType(), nullable=False)
    .add(CASHTAG_COL, psqlt.StringType(), nullable=False)
)
"""
JSON dataset schema, without *any* form of preprocessing. 
Schema may very well very after preprocessing.
"""


def download_dataset(return_train_dataset: bool) -> Path:
    train_dataset_path = _DOWNLOAD_DIR / 'Microblog_Trainingdata.json'
    test_dataset_path = _DOWNLOAD_DIR / 'Microblog_Testdata.json'
    dataset_path = train_dataset_path if return_train_dataset else test_dataset_path
    if not os.path.exists(dataset_path):
        logger.info('Downloading dataset...')
        subprocess.Popen(_DOWNLOAD_BASH_COMMAND, stdout=subprocess.PIPE, shell=True, executable="/bin/bash").communicate()
    else:
        logger.info('Dataset already downloaded')

    return dataset_path


def read_dataset(
        spark: psql.SparkSession,
        path: Path,
) -> psql.DataFrame:
    df = spark.read.option("multiline", True).json(str(path))

    return df


def clean_dataset(df: psql.DataFrame) -> psql.DataFrame:
    df = df.fillna({TEXT_COL: "", LABEL_COL: 1})  # 1 is neutral label in raw dataset

    # sentiment score column is string but we want float
    df = df.withColumn(LABEL_COL, psqlf.col(LABEL_COL).cast(psqlt.FloatType()))

    # Merge all spans into one text string
    @psqlf.udf(returnType=psqlt.StringType())
    def join_spans(spans: typing.List[str]) -> str:
        return " ".join(spans)

    df = df.withColumn(TEXT_COL, join_spans(psqlf.col(TEXT_COL)))

    # Drop useless cols
    df = df.drop(SOURCE_COL, CASHTAG_COL)

    return df
