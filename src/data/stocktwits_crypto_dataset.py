import os
import urllib.request
from pathlib import Path

import pandas as pd
import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf

import utils.io as io_


_DOWNLOAD_URL = 'https://huggingface.co/datasets/ElKulako/stocktwits-crypto/resolve/main/st-data-full.xlsx?download=true'

# This is the maximum number of characters of the texts in the final test dataset (SemEval)
# Assuming one token per character, we have a maximum of 160 tokens,
#   meaning that I'd throw away the remaining characters/tokens
#   so that memory and training times do not explode
WORST_CASE_TOKENS = 160

TEXT_COL = "text"
LABEL_COL = "label"

RAW_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
)


def download_dataset() -> Path:
    raw_xlsx_path = io_.RAW_DATASET_DIR / 'stocktwits-crypto.xlsx'
    raw_csv_path = io_.RAW_DATASET_DIR / 'stocktwits-crypto.csv'

    if not os.path.exists(raw_xlsx_path) or not os.path.exists(raw_csv_path):
        url = _DOWNLOAD_URL
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


def read_dataset(
        spark: psql.SparkSession,
        path: Path,
) -> psql.DataFrame:
    raw_docs_df: psql.DataFrame = spark.read.csv(str(path), header=True, schema=RAW_SCHEMA)

    return raw_docs_df


def clean(
        raw_df: psql.DataFrame,
        drop_neutral_samples: bool,
        text_col: str,
        label_col: str
) -> psql.DataFrame:
    raw_df = raw_df.fillna({text_col: "", label_col: 1})  # 1 is neutral label in raw dataset

    if drop_neutral_samples:
        # Drop neutral labels because they add noise:
        #   the absence of label is what defines them as neutral,
        #   meaning that they could in actuality express positive or negative.
        # Neutral label may hence prove misleading
        raw_df = raw_df.filter(f"{label_col} <> 1")

    return raw_df


def convert_labels_to_sentiment_scores(
        df: psql.DataFrame,
        label_col: str = LABEL_COL
) -> psql.DataFrame:
    @psqlf.udf(returnType=psqlt.FloatType())
    def convert_label(label: int) -> float:
        match label:
            case 0.0: return -1.0
            case 1.0: return 0.0
            case 2.0: return 1.0
            case _: raise ValueError(f'Unknown label {label}')

    with_sent_score_df = df.withColumn(label_col, convert_label(psqlf.col(label_col)))

    return with_sent_score_df
