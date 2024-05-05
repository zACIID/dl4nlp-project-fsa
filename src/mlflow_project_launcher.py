import datetime

import mlflow

import utils.io as io_
import utils.mlflow_env as env

if __name__ == "__main__":
    # NOTE: mlflow *should* read from env, but you never know...
    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    mlflow.projects.run(
        uri=str(io_.PROJECT_ROOT.absolute()),
        entry_point='hyperopt',
        env_manager='local',
        experiment_name=env.MLFLOW_EXPERIMENT_NAME,
        run_name=f"{datetime.date.today().isoformat()}-hyperopt",
        synchronous=True
    )
