import os
import typing
import urllib.request
from pathlib import Path

import click
import datasets
import pandas as pd
import pyspark.sql as psql
from loguru import logger
from pyspark.sql import types as psqlt, functions as psqlf
from transformers import AutoTokenizer, BatchEncoding

import data.spark as S
import data.stocktwits_crypto_dataset as sc
import utils.io as io_


_MODEL_NAME = 'end_to_end'
_SPARK_APP_NAME = f'{_MODEL_NAME} Preprocessing'

# TODO IMPORTANT: maybe I can just get away with stuff in the datamodule that uses the existing datamodules and return their combined batches
#   check stuff like Pytorch ConcatDataset or ChainingDataset, maybe it is useful
#   being able to combine existing dataloaders is best because it means that there is no data duplication
#   or maybe I just have to rewrite the datamodule almost from scracth but I have nonetheless the two processed datasets to combine.
#   maybe in the DataModule setup I can use some function of the datasets library to concatenate the two datasets on the rows
#   and then I am good
