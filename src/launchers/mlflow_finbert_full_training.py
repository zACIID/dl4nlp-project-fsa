import datetime

import mlflow

import utils.io as io_
import utils.mlflow_env as env
import training.loader as loader


if __name__ == "__main__":
    env.set_experiment_name_prefix(env.FULL_TRAINING_EXPERIMENT_NAME_PREFIX)

    # Complete training of models tuned on this train+val combo
    env.set_dataset_choice(loader.Dataset.SEMEVAL_TRAIN_VAL)
    env.set_model_choice(loader.Model.FINBERT)

    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    mlflow.projects.run(
        uri=str(io_.PROJECT_ROOT.absolute()),
        entry_point='semeval_full_training',
        env_manager='local',
        experiment_name=env.get_experiment_name(),
        run_name=f"{datetime.datetime.now().isoformat(timespec='seconds')}",
        parameters={
            'model_name': env.get_registered_model_name(env.get_model_choice()),
            'model_alias': env.get_dataset_specific_best_model_alias(env.get_dataset_choice(), tuning=True),
        },
        synchronous=True
    )
