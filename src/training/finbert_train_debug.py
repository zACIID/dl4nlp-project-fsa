import os
import datetime
import logging

import click
import mlflow
import mlflow.utils.autologging_utils
import dotenv
import lightning as L
import lightning.pytorch.callbacks as cb
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.profilers import SimpleProfiler

import data.fine_tuned_finbert.data_modules as dm
from models.fine_tuned_finbert import FineTunedFinBERT
from utils.io import PROJECT_ROOT
from utils.random import RND_SEED


RUN_NAME_PREFIX = 'finbert'

# NOTE: these defaults are for debug purposes
@click.command(
    help="Fine-tune FinBERT model"
)
@click.option("--train-batch-size", default=32, type=click.INT)
@click.option("--eval-batch-size", default=16, type=click.INT)
@click.option("--train-split-size", default=0.8, type=click.FLOAT)
@click.option("--prefetch-factor", default=16, type=click.INT)
@click.option("--num-workers", default=8, type=click.INT)
@click.option("--one-cycle-max-lr", default=1e-3, type=click.FLOAT)
@click.option("--one-cycle-pct-start", default=0.3, type=click.FLOAT)
@click.option("--weight-decay", default=1e-2, type=click.FLOAT)
@click.option("--lora-rank", default=150, type=click.INT)
@click.option("--lora-alpha", default=1.25, type=click.FLOAT)
@click.option("--max-epochs", default=25, type=click.INT)
@click.option("--accumulate-grad-batches", default=6, type=click.INT)
@click.option("--limit-batches", default=0.005, type=click.FLOAT)
@click.option("--es-monitor", default='val_loss', type=click.STRING)
@click.option("--es-min-delta", default=1e-3, type=click.FLOAT)
@click.option("--es-patience", default=500, type=click.INT)
@click.option("--ckpt-monitor", default='val_loss', type=click.STRING)
@click.option("--ckpt-save-top-k", default=10, type=click.INT)
def run(
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
        max_epochs,
        accumulate_grad_batches,
        limit_batches,
        es_monitor,
        es_min_delta,
        es_patience,
        ckpt_monitor,
        ckpt_save_top_k
):
    # configure logging at the root level of Lightning
    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, 'mlflow.env'))

    # NOTE commenting because it hangs at the beginning of training due to some bug when logging hparams
    #   also, for the same reason (endpoint log-batch broken of mlflow rest API), self.log_dict does not log anything
    # mlflow.pytorch.autolog()
    hparams_string = _get_name_from_hparams(**{
        "dataset_frac": limit_batches,
        "batch_size": train_batch_size,
        "grad_acc": accumulate_grad_batches,
        "oc_max_lr": one_cycle_max_lr,
        "oc_pct_start": one_cycle_pct_start,
        "wd": weight_decay,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    })
    with mlflow.start_run(
            log_system_metrics=True,
            run_name=f"{datetime.date.today().isoformat()}-{RUN_NAME_PREFIX}-{hparams_string}"
    ) as run:
        mlflow_logger = MLFlowLogger(
            # This should be by default (check MLFlowLogger source code),
            #   but apparently it doesn't correctly read the uri from the env
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            log_model=True,
            run_id=run.info.run_id
        )

        L.seed_everything(RND_SEED)

        data_module = dm.StocktwitsCryptoTrainVal(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            train_split_size=train_split_size,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            num_workers=num_workers,
            rnd_seed=RND_SEED
        )

        model = FineTunedFinBERT(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            one_cycle_max_lr=one_cycle_max_lr,
            one_cycle_pct_start=one_cycle_pct_start,
            weight_decay=weight_decay
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
            # log_every_n_steps=1,

            # Checkpoints are automatically logged to mlflow according to the ModelCheckpoint callback
            # Remember to call mlflow.pytorch.autolog() and set log_model=True in MLflowLogger
            # enable_checkpointing=True,
            accumulate_grad_batches=accumulate_grad_batches,

            limit_train_batches=limit_batches,
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


def _get_name_from_hparams(**hparams: dict) -> str:
    return "-".join([f"{k}={v}" for k, v in hparams.items()])


if __name__ == '__main__':
    run()
