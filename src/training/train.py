import datetime
import logging

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
@click.option("--train-split-size", default=0.9, type=click.FLOAT)
@click.option("--prefetch-factor", default=16, type=click.INT)
@click.option("--num-workers", default=8, type=click.INT)
@click.option("--one-cycle-max-lr", default=1e-5, type=click.FLOAT)
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

    env.set_common_run_tags(with_neutral_samples=with_neutral_samples)

    # configure logging at the root level of Lightning
    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    if env.get_dataset_choice() == loader.Dataset.SEMEVAL_TEST:
        raise ValueError(f'{env.get_dataset_choice()} is not a valid dataset choice for training/validation')

    mlflow.pytorch.autolog(
        checkpoint_monitor='val_loss',
        checkpoint_mode='min',
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=True,
        checkpoint_save_freq='epoch'
    )

    with mlflow.start_run(
        log_system_metrics=True,
        run_name=f"{datetime.datetime.now().isoformat(timespec='seconds')}-{env.get_model_choice()}",
        tags=env.get_run_tags()
    ) as run:
        L.seed_everything(RND_SEED)

        function_call_kwargs['rnd_seed'] = RND_SEED
        model, data_module = loader.get_model_and_data_module(
            model_choice=env.get_model_choice(),
            dataset_choice=env.get_dataset_choice(),
            model_init_args=function_call_kwargs,
            dm_init_args=function_call_kwargs
        )

        trainer = L.Trainer(
            default_root_dir=os.path.join(PROJECT_ROOT, "artifacts"),
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,
            profiler=SimpleProfiler(filename='simple-profiler-logs'),
            accumulate_grad_batches=accumulate_grad_batches,
            limit_train_batches=limit_batches,
            limit_val_batches=1.0,
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

        trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    run()
