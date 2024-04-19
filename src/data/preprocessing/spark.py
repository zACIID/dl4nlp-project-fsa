import json
import math
import sys
import os
from typing import Tuple, Type, List, Dict, Any

import pandas as pd
import pyspark.sql as psql
from loguru import logger
from matplotlib import pyplot as plt
from pyspark import SparkFiles

import utils.io as io_
from utils.system import MAX_AVAILABLE_CORES, MAX_AVAILABLE_RAM_GB

# Since the datasets used are relatively small, one partition per core is sufficient
# The rule of thumb would be "numPartitions = numWorkers * cpuCoresPerWorker"
# In my case, local standalone cluster, there is just 1 worker with AVAILABLE_CORES cores
# See this answer for a useful discussion about how to determine numPartitions
#   https://stackoverflow.com/a/39398750/19582401
N_PARTITIONS = MAX_AVAILABLE_CORES

# Needed to correctly set the python executable of the current conda env
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP'] = "127.0.0.1"

# UserWarning: 'PYARROW_IGNORE_TIMEZONE' environment variable was not set.
# It is required to set this environment variable to '1' in both driver and executor
#   sides if you use pyarrow>=2.0.0.
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# TODO all of the stuff above is actually fine to leave there as the module is loaded
# TODO maybe inject via spark.env file?
#   I'd like to specify the following:
#   - master host
#   - driver ram gb
#   - driver cores
#   - n executors
#   - cores available to executors
#   - ram available to executors
#   the last two default to max system cores and ram if not specified, else the resources
#   will be split equally among executors
MASTER_HOST = "localhost"  # master host in local standalone cluster

# TODO example usage
# Create "single-use" spark session to parse the text
# spark = create_spark_session(n_executors=4, app_name="Doc Features")
# docs_scores_df = tok.get_document_features(
#     spark=spark,
#     corpus_json_path=corpus_paths[d_name],
#     n_partitions=N_PARTITIONS
# )
# docs_scores_pandas: pd.DataFrame = docs_scores_df.toPandas()
# spark.stop()
def create_spark_session(app_name: str) -> psql.SparkSession:
    driver_ram_gb = 2
    driver_cores = 2
    executors_tot_ram = None TODO
    executors_tot_cores = None |TODo
    n_executors = None | TOdo
    mem_per_executor = executors_tot_ram // n_executors
    cores_per_executor = executors_tot_cores // n_executors

    logger.debug(f"Executor memory: {mem_per_executor}")
    logger.debug(f"AVAILABLE_RAM_GB: {MAX_AVAILABLE_RAM_GB}")
    logger.debug(f"Total executor memory: {(MAX_AVAILABLE_RAM_GB - driver_ram_gb)}")
    logger.debug(f"Executor cores: {cores_per_executor}")


    spark: psql.SparkSession = (
        psql.SparkSession.builder
        .master(f"spark://{MASTER_HOST}:7077")  # connect to previously started master host

        .appName(f"{app_name}")
        #.config("spark.driver.host", f"{MASTER_HOST}:7077")
        .config("spark.driver.cores", driver_cores)
        .config("spark.driver.memory", f"{driver_ram_gb}g")
        .config("spark.executor.instances", n_executors)
        .config("spark.executor.cores", cores_per_executor)
        .config("spark.executor.memory", f"{mem_per_executor}g")
        .config("spark.default.parallelism", MAX_AVAILABLE_CORES)
        .config("spark.cores.max", MAX_AVAILABLE_CORES - driver_cores)
        .getOrCreate()
    )

    # Add local dependencies (local python source files) to SparkContext and sys.path
    src_zip_path = os.path.abspath(io_.SRC_DIR)
    logger.debug(f"Adding {src_zip_path} to SparkContext")

    spark.sparkContext.addPyFile(src_zip_path)
    sys.path.insert(0, SparkFiles.getRootDirectory())

    return spark

