import os
import datetime

import dotenv
import mlflow

import utils.io as io_

dotenv.load_dotenv(str(io_.PROJECT_ROOT / 'mlflow.env'))

# NOTE: mlflow *should* read from env, but you never know...
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.projects.run(
    uri=str(io_.PROJECT_ROOT.absolute()),
    entry_point='hyperopt',
    env_manager='local',
    experiment_name=os.environ['MLFLOW_EXPERIMENT_NAME_CUSTOM'],  # TODO apparently setting MLFLOW_EXPERIMENT_NAME breaks the mlflow.projects.run API
    run_name=f"{datetime.date.today().isoformat()}-hyperopt",
    synchronous=True
)
