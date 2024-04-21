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

import data.preprocessing.datasets.stocktwits_crypto as sc
from data.data_modules.finbert_train_val_data_module import FinBERTTrainValDataModule
from models.fine_tuned_finbert import FineTunedFinBERT
from utils.io import PROJECT_ROOT
from utils.random import RND_SEED


RUN_NAME_PREFIX = 'finbert'  # TODO parameterize based on model type?

# TODO this script could be generalized with a parameter "model-type" and a little switch that
#  chooses the classes of model, data module to instantiate.
#  Since each model/data module has (must havae) a **kwargs argument, I can directly pass everything and parameters
#   that a model doesn't care about will be auto ignored
#   (self.save_hparams I think has an option to ignore stuff, there I can ignore "**kwargs:
@click.command(
    help="Fine-tune FinBERT model"
)
@click.option("--train-batch-size", type=click.INT)
@click.option("--eval-batch-size", type=click.INT)
@click.option("--train-split-size", type=click.FLOAT)
@click.option("--prefetch-factor", type=click.INT)
@click.option("--num-workers", type=click.INT)
@click.option("--one-cycle-max-lr", type=click.FLOAT)
@click.option("--one-cycle-pct-start", type=click.FLOAT)
@click.option("--weight-decay", type=click.FLOAT)
@click.option("--lora-rank", type=click.INT)
@click.option("--lora-alpha", type=click.FLOAT)
@click.option("--max-epochs", type=click.INT)
@click.option("--accumulate-grad-batches", type=click.INT)
@click.option("--limit-batches", type=click.FLOAT)
@click.option("--es-monitor", type=click.STRING)
@click.option("--es-min-delta", type=click.FLOAT)
@click.option("--es-patience", type=click.INT)
@click.option("--ckpt-monitor", type=click.STRING)
@click.option("--ckpt-save-top-k", type=click.INT)
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
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, 'mlflow.env'))

    # TODO commenting because it hangs at the beginning of training due to some bug when logging hparams
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

        dm = FinBERTTrainValDataModule(
            dataset=sc.get_dataset(),
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

        trainer.fit(model, datamodule=dm)

        # TODO do something with best model? maybe register to mlflow model registry??
        # ckpt_callback.best_model_path
        # result = mlflow.register_model(
        #     ckpt_callback.best_model_path,
        #     "ElasticNetWineModel"
        # )

def _get_name_from_hparams(**hparams: dict) -> str:
    return "-".join([f"{k}={v}" for k, v in hparams.items()])


if __name__ == '__main__':
    run()
