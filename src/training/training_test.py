# TODO here I should define three training scripts, one for each model, that are based on the hyperopt mlflow example
#   I think I can gather them all inside one training module and add a base training_utils.py module so that I can
#   reuse the hyperopt code and just handle parameters in the various scripts
# TODO 2: define the MLflow project thing, along with docker image and env, and the three training entrypoints, one per
#   training script
# TODO entry points to define:
# - debug entry point with debug stuff on the Trainer class, as highlighted in this guide:
#   - SHORT RUNS to verify everything ok: https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html
#   - BASIC PROFILING: https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html

import os
import datetime
import logging

import mlflow
import dotenv
import lightning as L
import lightning.pytorch.callbacks as cb
from lightning.pytorch.loggers import MLFlowLogger
import mlflow.utils.autologging_utils
import mlflow.utils.autologging_utils.client
from lightning.pytorch.profilers import SimpleProfiler

import data.preprocessing.datasets.stocktwits_crypto as sc
from data.data_modules.finbert_train_val_data_module import FinBERTTrainValDataModule
from models.fine_tuned_finbert import FineTunedFinBERT
from utils.io import PROJECT_ROOT
from utils.random import RND_SEED

EXPERIMENT_NAME = 'Training Test'
RUN_NAME_PREFIX = 'training_test'


# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, 'mlflow.env'))

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow.pytorch.autolog()

    with mlflow.start_run(
            log_system_metrics=True,
            run_name=f"{RUN_NAME_PREFIX}-{datetime.datetime.now()}"
    ) as run:
        mlflow_logger = MLFlowLogger(
            # This should be by default (check MLFlowLogger source code),
            #   but apparently it doesn't correctly read the uri from the env
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            experiment_name=EXPERIMENT_NAME,
            log_model=True,
            run_id=run.info.run_id
        )

        L.seed_everything(RND_SEED)

        dm = FinBERTTrainValDataModule(
            dataset=sc.get_dataset(),
            train_batch_size=64,
            eval_batch_size=8,
            train_split_size=0.8,
            prefetch_factor=8,
            pin_memory=True,
            num_workers=6,
            rnd_seed=RND_SEED
        )

        model = FineTunedFinBERT(
            lora_rank=8,
            enable_gradient_checkpointing=False # TODO this will be remvoed
        )

        max_epochs = 10
        ckpt_callback = cb.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
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
            accumulate_grad_batches=4,
            precision='16-mixed',
            callbacks=[
                ckpt_callback,
                cb.EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-4,
                    patience=2,
                    strict=True
                ),
                cb.LearningRateMonitor(logging_interval='epoch'),
            ],
        )

        trainer.fit(model, datamodule=dm)
        # TODO do something with best model, maybe register to mlflow model registry
        # ckpt_callback.best_model_path
        # result = mlflow.register_model(
        #     ckpt_callback.best_model_path,
        #     "ElasticNetWineModel"
        # )


