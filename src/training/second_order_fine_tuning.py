import datetime
import logging
import os

import click
import lightning as L
import lightning.pytorch.callbacks as cb
import mlflow
import mlflow.utils.autologging_utils
from lightning.pytorch.profilers import SimpleProfiler

import training.loader as loader
import utils.mlflow_env as env
from utils.io import PROJECT_ROOT, ARTIFACTS_DIR
from utils.random import RND_SEED


# NOTE: these defaults are for debug purposes
@click.command(
    help=f"Fine-tune model already trained on the {loader.Dataset.SC_TRAIN_SEMEVAL_VAL} dataset"
)
@click.option("--train-batch-size", default=16, type=click.INT)
@click.option("--eval-batch-size", default=16, type=click.INT)
@click.option("--train-split-size", default=0.9, type=click.FLOAT)
@click.option("--prefetch-factor", default=16, type=click.INT)
@click.option("--num-workers", default=8, type=click.INT)
@click.option("--max-epochs", default=150, type=click.INT)
@click.option("--accumulate-grad-batches", default=1, type=click.INT)
@click.option("--limit-batches", default=1.0, type=click.FLOAT)
@click.option("--es-monitor", default='val_loss', type=click.STRING)
@click.option("--es-min-delta", default=1e-3, type=click.FLOAT)
@click.option("--es-patience", default=500, type=click.INT)
@click.option("--ckpt-monitor", default='val_loss', type=click.STRING)
@click.option("--ckpt-save-top-k", default=1, type=click.INT)
def train(
        train_batch_size,
        eval_batch_size,
        train_split_size,
        prefetch_factor,
        num_workers,
        max_epochs,
        accumulate_grad_batches,
        limit_batches,
        es_monitor,
        es_min_delta,
        es_patience,
        ckpt_monitor,
        ckpt_save_top_k
):
    function_call_kwargs = locals()

    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    mlflow.pytorch.autolog(
        checkpoint_monitor=ckpt_monitor,
        checkpoint_mode='min',
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=True,
        checkpoint_save_freq='epoch'
    )

    with mlflow.start_run(
        log_system_metrics=True,
        run_name=f"{datetime.datetime.now().isoformat(timespec='seconds')}-2nd-order-fine-tuning",
        tags=env.get_run_tags()
    ) as run:
        L.seed_everything(RND_SEED)

        dataset_choice = loader.Dataset.SEMEVAL_TRAIN_VAL

        function_call_kwargs['rnd_seed'] = RND_SEED
        virgin_model, data_module = loader.get_model_and_data_module(
            model_choice=env.get_model_choice(),
            dataset_choice=dataset_choice,
            model_init_args={},  # ignored because the real model is instantiated later
            dm_init_args=function_call_kwargs
        )

        # Even if there are two SC_TRAIN_* datasets, I am using just the one that uses the SEMEVAL_VAL
        #   because its loss is more interpretable in terms of likelihood of performing well with SEMEVAL_TRAIN_VAL
        model_name = env.get_registered_model_name(env.get_model_choice())
        alias = env.get_dataset_specific_best_model_alias(dataset=loader.Dataset.SC_TRAIN_SEMEVAL_VAL, tuning=True)

        # If this fails then ok, a model must be trained on the above dataset
        #   for this 2nd order fine-tuning to make sense
        version = client.get_model_version_by_alias(
            name=model_name,
            alias=alias
        )
        logging.info(f"Found version for alias '{alias}': {version.version}")

        best_model = mlflow.pytorch.load_checkpoint(
            virgin_model.__class__,
            version.run_id,
            kwargs={
                'strict': False,  # `True` does not work with FinBERT, because checkpoints contain only LoRA weights
                'log_hparams': False  # autolog is already active, setting to True causes problems
            }
        )
        del virgin_model

        trainer = L.Trainer(
            default_root_dir=ARTIFACTS_DIR,
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,
            profiler=SimpleProfiler(filename='simple-profiler-logs'),
            accumulate_grad_batches=accumulate_grad_batches,
            limit_train_batches=limit_batches,
            limit_val_batches=limit_batches,
            precision='16-mixed',
            callbacks=[
                cb.EarlyStopping(
                    monitor=es_monitor,
                    min_delta=es_min_delta,
                    patience=es_patience,
                    strict=True
                ),
                cb.LearningRateMonitor(logging_interval='epoch'),
            ],
        )

        trainer.fit(best_model, datamodule=data_module)


if __name__ == '__main__':
    train()
