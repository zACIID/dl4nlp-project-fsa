import datetime

import mlflow

import utils.io as io_
import utils.mlflow_env as env
import training.loader as loader

if __name__ == "__main__":
    env.set_experiment_name_prefix(env.HYPEROPT_EXPERIMENT_NAME_PREFIX)
    env.set_dataset_choice(loader.Dataset.SEMEVAL_TRAIN_VAL)
    env.set_model_choice(loader.Model.FINBERT)
    env.set_hyperopt_on_pretrained_model()  # Start with the weights obtained after pre-training on SC dataset

    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    mlflow.projects.run(
        uri=str(io_.PROJECT_ROOT.absolute()),
        entry_point='hyperopt_finbert',
        env_manager='local',
        experiment_name=env.get_experiment_name(),
        run_name=f"{datetime.datetime.now().isoformat(timespec='seconds')}",
        parameters={
            # Use everything when dataset is SEMEVAL_TRAIN_VAL,
            # else use much smaller part if SC_TRAIN_SEMEVAL_VAL (else 1.3Mln total, 4h per epoch)
            'limit_batches': 1.0 if env.get_dataset_choice() == loader.Dataset.SEMEVAL_TRAIN_VAL else 0.015,

            # Less runs if pre-training with SC dataset, which has longer epochs
            'max_runs': 70 if env.get_dataset_choice() == loader.Dataset.SEMEVAL_TRAIN_VAL else 35
        },
        synchronous=True
    )
