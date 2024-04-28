import shutil
import sys
import os
from pathlib import Path

import dotenv
import pyspark.sql as psql
from loguru import logger
from pyspark import SparkFiles

import utils.io as io_


dotenv.load_dotenv(io_.PROJECT_ROOT / 'spark' / 'spark.application.env')

MASTER_HOST = os.getenv('MASTER_HOST')
DRIVER_RAM_GB = int(os.getenv('DRIVER_RAM_GB'))
DRIVER_CORES = int(os.getenv('DRIVER_CORES'))
N_EXECUTORS = int(os.getenv('N_EXECUTORS'))
EXECUTORS_AVAILABLE_RAM_GB = int(os.getenv('EXECUTORS_AVAILABLE_RAM_GB'))
EXECUTORS_AVAILABLE_CORES = int(os.getenv('EXECUTORS_AVAILABLE_CORES'))

# Since the datasets used are relatively small, one partition per core is sufficient
# The rule of thumb would be "numPartitions = numWorkers * cpuCoresPerWorker"
# In my case, local standalone cluster, there is just 1 worker with AVAILABLE_CORES cores
# See this answer for a useful discussion about how to determine numPartitions
#   https://stackoverflow.com/a/39398750/19582401
N_PARTITIONS = EXECUTORS_AVAILABLE_CORES


# See spark-setup.md to understand why this is needed
os.environ['PYSPARK_PYTHON'] = str(io_.PROJECT_ROOT / '.venv' / 'bin' / 'python')


def create_spark_session(app_name: str) -> psql.SparkSession:
    logger.info(f'Creating spark session for {app_name}')

    mem_per_executor = EXECUTORS_AVAILABLE_RAM_GB // N_EXECUTORS
    cores_per_executor = EXECUTORS_AVAILABLE_CORES // N_EXECUTORS

    logger.debug(f"RAM GB/Executor: {mem_per_executor}")
    logger.debug(f"Cores/Executor: {cores_per_executor}")
    logger.debug(f"Total executor RAM GB: {EXECUTORS_AVAILABLE_RAM_GB}")

    spark: psql.SparkSession = (
        psql.SparkSession.builder
        .master(f"spark://{MASTER_HOST}:7077")  # connect to previously started master host

        .appName(f"{app_name}")
        #.config("spark.driver.host", f"{MASTER_HOST}:7077")
        .config("spark.driver.cores", DRIVER_CORES)
        .config("spark.driver.memory", f"{DRIVER_RAM_GB}g")
        .config("spark.executor.instances", N_EXECUTORS)
        .config("spark.executor.cores", cores_per_executor)
        .config("spark.executor.memory", f"{mem_per_executor}g")
        # TODO not sure if needed, check https://spark.apache.org/docs/latest/configuration.html
        #   I think this determines the number of RDD partitions
        .config("spark.default.parallelism", EXECUTORS_AVAILABLE_CORES)
        .config("spark.cores.max", EXECUTORS_AVAILABLE_CORES)
        .getOrCreate()
    )

    # Add local dependencies (local python source files) to SparkContext and sys.path
    src_zip_path = _ensure_src_dir_is_zipped()
    logger.debug(f"Adding {src_zip_path} to SparkContext")

    spark.sparkContext.addPyFile(str(src_zip_path.absolute()))
    sys.path.insert(0, SparkFiles.getRootDirectory())

    return spark


def _ensure_src_dir_is_zipped() -> Path:
    src_zip_path = io_.PROJECT_ROOT / 'spark' / 'src.zip'
    logger.debug(f"Zipping `{io_.SRC_DIR}` directory to {src_zip_path}")
    shutil.make_archive(io_.PROJECT_ROOT / 'spark' / 'src', 'zip', root_dir=io_.SRC_DIR, base_dir=io_.SRC_DIR, verbose=True)

    return src_zip_path
