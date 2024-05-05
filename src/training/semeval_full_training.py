import datetime
import logging
from collections import ChainMap

import click
import lightning as L
import lightning.pytorch.callbacks as cb
import mlflow
import mlflow.utils.autologging_utils
from lightning.pytorch.profilers import SimpleProfiler
from mlflow import MlflowClient

import training.loader as loader
import utils.mlflow_env as env
from utils.io import ARTIFACTS_DIR
from utils.random import RND_SEED


# NOTE: these defaults are for debug purposes
@click.command(
    help=f"Train a specified model (by name, alias) on the full SemEval 2017 Task 5 SubTask 1 training dataset"
)
@click.option("--model-name", default=env.get_registered_model_name(loader.Model.FINBERT), type=click.STRING)
@click.option("--model-alias", default=env.BEST_TUNED_MODEL_ALIAS, type=click.STRING)
@click.option("--prefetch-factor", default=16, type=click.INT)
@click.option("--num-workers", default=8, type=click.INT)
def train(
        model_name,
        model_alias,
        prefetch_factor,
        num_workers,
):
    function_call_kwargs = locals()

    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    mlflow.pytorch.autolog(
        checkpoint_monitor='train_loss',  # I am ok with saving just the last epochs here
        checkpoint_mode='min',
        checkpoint_save_best_only=True,
        # Since checkpointing here is purely save-to-resume-in-case-of-error, save everything
        checkpoint_save_weights_only=False,
        checkpoint_save_freq='epoch'
    )

    with mlflow.start_run(
        log_system_metrics=True,
        run_name=f"{datetime.datetime.now().isoformat(timespec='seconds')}-full-training",
        tags=env.get_run_tags()
    ) as run:
        L.seed_everything(RND_SEED)
        function_call_kwargs['rnd_seed'] = RND_SEED

        # If this fails then ok, a model must be trained on the above dataset
        #   for this 2nd order fine-tuning to make sense
        version = client.get_model_version_by_alias(
            name=model_name,
            alias=model_alias
        )
        logging.info(f"Found version for alias '{model_alias}': {version.version}")

        client = MlflowClient()
        model_run = client.get_run(version.run_id)

        # Merge current params with what was used for the model run
        # Maps ordered from first-searched to last-searched - give priority to overrides from current run
        dm_init_args = ChainMap(
            function_call_kwargs,

            # Load just some specific, datamodule-related params from the best run
            # Fine-tuned params are stored with the model
            {
                'train_batch_size': int(model_run.data.params['train_batch_size']),
                'accumulate_grad_batches': int(model_run.data.params['accumulate_grad_batches'])
            }
        )

        semeval_train_dataset = loader.Dataset.SEMEVAL_TRAIN
        virgin_model, data_module = loader.get_model_and_data_module(
            model_choice=env.get_model_choice(),
            dataset_choice=semeval_train_dataset,
            model_init_args={
                'log_hparams': False  # see comment below
            },  # ignored because the real model is instantiated later
            dm_init_args=dm_init_args
        )

        best_model: L.LightningModule = mlflow.pytorch.load_checkpoint(
            virgin_model.__class__,
            version.run_id,
            kwargs={
                'strict': False,  # `True` does not work with FinBERT, because checkpoints contain only LoRA weights
                'log_hparams': False  # autolog is already active, setting to True causes problems
            }
        )
        best_params = best_model.hparams
        del best_model  # Not needed anymore, just needed tuned hparams
        virgin_model: L.LightningModule = virgin_model.__class__(**best_params)

        trainer = L.Trainer(
            default_root_dir=ARTIFACTS_DIR,
            max_epochs=int(model_run.data.params['epochs']),  # train for the exact number of epochs of the tuned model
            accelerator="gpu",
            devices=1,
            profiler=SimpleProfiler(filename='simple-profiler-logs'),
            accumulate_grad_batches=dm_init_args['accumulate_grad_batches'],
            limit_train_batches=1.0,
            limit_val_batches=0.0,  # Disable validation
            precision='16-mixed',
            callbacks=[
                cb.LearningRateMonitor(logging_interval='epoch'),
            ],
        )

        # The goal is to retrain a virgin model on the full dataset with the best set of params
        trainer.fit(virgin_model, datamodule=data_module)

        # Add (overwrite) best model to registry
        version = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/artifacts/model",
            name=model_name,
            tags=env.get_run_tags()
        )
        client.set_registered_model_alias(
            name=model_name,
            alias=env.BEST_FULL_TRAINED_MODEL_ALIAS,
            version=version.version
        )
        client.set_registered_model_alias(
            name=model_name,
            alias=env.get_dataset_specific_best_model_alias(dataset=env.get_dataset_choice(), tuning=False),
            version=version.version
        )


if __name__ == '__main__':
    train()
