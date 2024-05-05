import datetime
import logging
import os

import click
import lightning as L
import lightning.pytorch.callbacks as cb
import mlflow
import mlflow.utils.autologging_utils
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.profilers import SimpleProfiler

import training.loader as loader
import utils.mlflow_env as env
from utils.io import PROJECT_ROOT
from utils.random import RND_SEED


# NOTE: these defaults are for debug purposes
@click.command(
    help="Fine-tune FinBERT model"
)
@click.option("--with-neutral-samples", default='false', type=click.STRING)
@click.option("--train-batch-size", default=32, type=click.INT)
@click.option("--eval-batch-size", default=16, type=click.INT)
@click.option("--train-split-size", default=0.8, type=click.FLOAT)
@click.option("--prefetch-factor", default=16, type=click.INT)
@click.option("--num-workers", default=8, type=click.INT)
@click.option("--one-cycle-max-lr", default=1e-3, type=click.FLOAT)
@click.option("--one-cycle-pct-start", default=0.3, type=click.FLOAT)
@click.option("--weight-decay", default=1e-5, type=click.FLOAT)
@click.option("--lora-rank", default=128, type=click.INT)
@click.option("--lora-alpha", default=1, type=click.FLOAT)
@click.option("--lora-dropout", default=0.1, type=click.FLOAT)
@click.option("--C", "C", default=0.1, type=click.FLOAT)
@click.option("--max-epochs", default=20, type=click.INT)
@click.option("--accumulate-grad-batches", default=4, type=click.INT)
# @click.option("--limit-batches", default=1, type=click.FLOAT)
@click.option("--limit-batches", default=0.003, type=click.FLOAT)
@click.option("--es-monitor", default='val_loss', type=click.STRING)
@click.option("--es-min-delta", default=1e-3, type=click.FLOAT)
@click.option("--es-patience", default=500, type=click.INT)
@click.option("--ckpt-monitor", default='val_loss', type=click.STRING)
@click.option("--ckpt-save-top-k", default=1, type=click.INT)
def run(
        with_neutral_samples,
        train_batch_size,
        eval_batch_size,
        train_split_size,
        prefetch_factor,
        num_workers,
        one_cycle_max_lr,
        one_cycle_pct_start,
        weight_decay,
        lora_rank,
        lora_alpha,
        lora_dropout,
        C,
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
    function_call_kwargs['with_neutral_samples'] = True if with_neutral_samples == 'true' else False

    # configure logging at the root level of Lightning
    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    if env.DATASET_CHOICE == loader.Dataset.SEMEVAL_TEST:
        raise ValueError(f'{env.DATASET_CHOICE} is not a valid dataset choice for training/validation')

    model_choice: loader.Model = loader.Model(model_choice)
    dataset_choice: loader.Dataset = loader.Dataset(dataset_choice)
    if dataset_choice == loader.Dataset.SEMEVAL_TEST:
        raise ValueError(f'{dataset_choice} is not a valid dataset choice for training/validation')

    run_name_prefix = f"{env.MODEL_CHOICE}_{env.DATASET_CHOICE}"
    with mlflow.start_run(
            log_system_metrics=True,
            run_name=f"{datetime.date.today().isoformat()}-{run_name_prefix}"
    ) as run:
        mlflow_logger = MLFlowLogger(
            # This should be by default (check MLFlowLogger source code),
            #   but apparently it doesn't correctly read the uri from the env
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            log_model=True,
            run_id=run.info.run_id
        )

        L.seed_everything(RND_SEED)

        function_call_kwargs['rnd_seed'] = RND_SEED
        model, data_module = loader.get_model_and_data_module(
            model_choice=env.MODEL_CHOICE,
            dataset_choice=env.DATASET_CHOICE,
            model_init_args=function_call_kwargs,
            dm_init_args=function_call_kwargs
        )

        ckpt_callback = cb.ModelCheckpoint(
            monitor=ckpt_monitor,
            mode='min',
            save_top_k=ckpt_save_top_k,
            save_weights_only=False
        )

        trainer = L.Trainer(
            default_root_dir=os.path.join(PROJECT_ROOT, "artifacts"),
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,
            profiler=SimpleProfiler(filename='simple-profiler-logs'),
            logger=mlflow_logger,

            # Checkpoints are automatically logged to mlflow according to the ModelCheckpoint callback
            # Remember to call mlflow.pytorch.autolog() and set log_model=True in MLflowLogger
            # enable_checkpointing=True,
            accumulate_grad_batches=accumulate_grad_batches,

            limit_train_batches=limit_batches,
            # Do not limit val batches if val dataset is small
            limit_val_batches=limit_batches,
            precision='16-mixed',
            callbacks=[
                ckpt_callback,
                cb.EarlyStopping(
                    monitor=es_monitor,
                    min_delta=es_min_delta,
                    patience=es_patience,
                    strict=True
                ),
                cb.LearningRateMonitor(logging_interval='epoch'),
            ],
        )

        trainer.fit(model, datamodule=data_module)

        # TODO do something with best model? maybe register to mlflow model registry??
        # ckpt_callback.best_model_path
        # mlflow.pytorch.log_model()
        # result = mlflow.register_model(
        #     ckpt_callback.best_model_path,
        #     "registration-test"
        # )


def _get_name_from_hparams(**hparams: dict) -> str:
    return "-".join([f"{k}={v}" for k, v in hparams.items()])


if __name__ == '__main__':
    run()
