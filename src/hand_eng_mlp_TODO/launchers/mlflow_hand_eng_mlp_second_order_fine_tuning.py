import datetime

import mlflow

import utils.io as io_
import utils.mlflow_env as env
import training.loader as loader


if __name__ == "__main__":
    env.set_experiment_name_prefix(env.FULL_TRAINING_EXPERIMENT_NAME_PREFIX)

    # Continue fine-tuning of models trained only on this dataset
    env.set_dataset_choice(loader.Dataset.SC_TRAIN_SEMEVAL_VAL)

    env.set_model_choice(loader.Model.HAND_ENG_MLP)

    # 'Second order fine-tuning' because we are training on a second dataset,
    #   from stocktwits-crypto to the semeval dataset
    env.set_common_run_tags(second_order_fine_tuning=True)

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
