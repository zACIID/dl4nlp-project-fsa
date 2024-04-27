import os
import datetime

import dotenv
import mlflow

import utils.io as io_

dotenv.load_dotenv(str(io_.PROJECT_ROOT / 'mlflow.env'))

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
experiment = mlflow.set_experiment('Hyperopt')
mlflow.projects.run(
    uri=str(io_.PROJECT_ROOT.absolute()),
    entry_point='hyperopt',
    env_manager='local',
    experiment_id=experiment.experiment_id,
    run_name=f"{datetime.date.today().isoformat()}-hyperopt",
    synchronous=True
)
