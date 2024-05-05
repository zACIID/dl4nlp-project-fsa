import datetime

import mlflow

import utils.io as io_
import utils.mlflow_env as env
import training.loader as loader


if __name__ == "__main__":
    env.set_experiment_name_prefix('2nd Order Fine Tuning')
    env.set_dataset_choice(loader.Dataset.SEMEVAL_TRAIN_VAL)
    env.set_model_choice(loader.Model.FINBERT)
    env.set_common_run_tags(second_order_fine_tuning=True)

    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    mlflow.projects.run(
        uri=str(io_.PROJECT_ROOT.absolute()),
        entry_point='second_order_fine_tuning',
        env_manager='local',
        experiment_name=env.get_experiment_name(),
        run_name=f"{datetime.datetime.now().isoformat(timespec='seconds')}",
        synchronous=True
    )
