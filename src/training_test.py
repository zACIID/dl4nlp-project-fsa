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

from src.data.train_val_data_module import TrainValDataModule
from src.models.fine_tuned_finbert import FineTunedFinBERT
from src.utils.utils import PROJECT_ROOT


if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, 'mlflow.env'))

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    mlflow.pytorch.autolog()

    EXPERIMENT_NAME = 'first test'
    # mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # TODO mlflow is not properly setup - check what I wrote in my notes and see some examples
    #  (e.g. I need to launch the script, call mlflow.set_experiment, mlflow.start_run, etc.)
    mlflow_logger = MLFlowLogger(
        # This should be by default (check MLFlowLogger source code),
        #   but apparently it doesn't correctly read the uri from the env
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
        experiment_name=EXPERIMENT_NAME
    )

    with mlflow.start_run(log_system_metrics=True) as run:
        L.seed_everything(42)

        dm = TrainValDataModule(
            train_batch_size=8,
            eval_batch_size=8,
            prefetch_factor=4,
            pin_memory=True,
            num_workers=16
        )
        dm.setup()

        max_epochs = 10

        # TODO check this for GRADIENT checkpointing and try to implement it in FineTunedFinBERT
        # https://residentmario.github.io/pytorch-training-performance-guide/gradient-checkpoints.html
        # Can I parameterize the number of steps after which checkpointing happens?
        # TODO 2: but FIRST SEE HOW IS MEMORY WITHOUT CHECKPOINTING

        model = FineTunedFinBERT(
            epochs=max_epochs,
            n_batches=len(dm.train_dataloader()),
            lora_rank=8,
            enable_gradient_checkpointing=False # TODO this will be remvoed
        )

        trainer = L.Trainer(
            default_root_dir=os.path.join(PROJECT_ROOT, "artifacts"),
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            profiler='simple',
            logger=mlflow_logger,

            # TODO idea here is to gradually try all of these parameters and see the impact on VRAM
            #   also try to use the basic profiling api and the short dev runs api, see urls at the beginning of file

            # Checkpoints are automatically logged to mlflow according to the ModelCheckpoint callback
            # Remember to call mlflow.pytorch.autolog()
            # TODO See Examples:
            # - https://www.restack.io/docs/mlflow-knowledge-mlflow-pytorch-lightning-integration
            # - https://dwarez.github.io/posts/lightning_and_mlflow_logging/
            enable_checkpointing=False,
            accumulate_grad_batches=1,  # TODO this doesn't work possibly becaused of pre trained model?
            # precision='16-mixed',
            # callbacks=[
            #     cb.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5),
            #     cb.ModelCheckpoint(
            #         monitor='val_loss',
            #         mode='min',
            #         save_top_k=3,
            #         save_weights_only=False
            #     ),
            # ],
        )

        trainer.fit(model, datamodule=dm)


