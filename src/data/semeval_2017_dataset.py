import os
import urllib.request
from pathlib import Path

import pandas as pd
from loguru import logger
from pyspark.sql import types as psqlt

import utils.io as io_


_DOWNLOAD_URL = 'https://huggingface.co/datasets/ElKulako/stocktwits-crypto/resolve/main/st-data-full.xlsx?download=true'

# This is the maximum number of characters of the texts in the final test dataset (SemEval)
# Assuming one token per character, we have a maximum of 160 tokens,
#   meaning that I'd throw away the remaining characters/tokens
#   so that memory and training times do not explode
WORST_CASE_TOKENS = 160

TEXT_COL = "text"
LABEL_COL = "label"

SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
    .add(LABEL_COL, psqlt.IntegerType(), nullable=False)
)


def download_dataset() -> Path:
    # TODO retrieve dataset from semeval2017 repo
    # TODO create preprocessing and datamodules for each model under a "semeval_2017" module
    # TODO gather the common preprocessing logic into a preprocessing_base.py script that is outside the dataset-specific modules
    #   TODO maybe even make a data/preprocessing_common module where I put every utils function that may be needed during cleaning by other preprocessing modules
    # TODO parameterize such logic so that different datasets can work on it: I just need to know the name of the text and label columns at the end of the day
    raise NotImplementedError('TODO')

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
