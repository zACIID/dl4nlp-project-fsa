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

import mlflow
import dotenv
import lightning as L
import lightning.pytorch.callbacks as cb
from lightning.pytorch.loggers import MLFlowLogger
import mlflow.utils.autologging_utils
from lightning.pytorch.profilers import SimpleProfiler

from data.train_val_data_module import TrainValDataModule
from models.fine_tuned_finbert import FineTunedFinBERT
from utils.utils import PROJECT_ROOT


if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, 'mlflow.env'))

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    mlflow.pytorch.autolog()

    EXPERIMENT_NAME = 'first test'
    # mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow_logger = MLFlowLogger(
        # This should be by default (check MLFlowLogger source code),
        #   but apparently it doesn't correctly read the uri from the env
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
        experiment_name=EXPERIMENT_NAME,
        log_model=True
    )

    with mlflow.start_run(log_system_metrics=True) as run:
        L.seed_everything(42)

        dm = TrainValDataModule(
            train_batch_size=64,
            eval_batch_size=8,
            prefetch_factor=4,
            pin_memory=True,
            num_workers=16
        )
        dm.setup()

        max_epochs = 10

        model = FineTunedFinBERT(
            lora_rank=8,
            enable_gradient_checkpointing=False # TODO this will be remvoed
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
            enable_checkpointing=True,
            accumulate_grad_batches=4,
            precision='16-mixed',
            callbacks=[
                cb.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=2),
                cb.ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=3,
                    save_weights_only=False
                ),
                cb.LearningRateMonitor(logging_interval='step'),
            ],
        )

        trainer.fit(model, datamodule=dm)


