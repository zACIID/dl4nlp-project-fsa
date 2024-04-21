"""
Reference:
https://github.dev/mlflow/mlflow/blob/master/examples/hyperparam/search_hyperopt.py
"""
import datetime
import math
import os
import sys
import typing

import click
import dotenv
import numpy as np
import hyperopt as hp
from hyperopt import fmin, hp, rand, tpe, Trials
from hyperopt.pyll import scope

import mlflow.projects
from mlflow.tracking import MlflowClient

import utils.io as io_

dotenv.load_dotenv(str(io_.PROJECT_ROOT / '..' / 'mlflow.env'))

_inf = np.finfo(np.float64).max

EXPERIMENT_NAME = 'hyperopt'
MODEL_NAME = 'finbert'  # TODO one day this will be a script parameter
MLFLOW_PROJECT_ENTRYPOINT = 'train'


def new_eval(experiment_id, max_epochs, limit_batches, fail_on_error: bool):
    def eval(params: typing.Dict[str, typing.Any]):
        """
        Train Keras model with given parameters by invoking MLflow run.

        Notice we store runUuid and resulting metric in a file. We will later use these to pick
        the best run and to log the runUuids of the child runs as an artifact. This is a
        temporary workaround until MLflow offers better mechanism of linking runs together.

        Args:
            params: Parameters to the train_keras script we optimize over:
                learning_rate, drop_out_1

        Returns:
            The metric value evaluated on the validation data.
        """
        tracking_client = MlflowClient()

        # These are params excluded from hparam search that still
        #   need to be passed to the training run
        params["max_epochs"] = max_epochs
        params["limit_batches"] = limit_batches

        hparams_suffix = '-'.join([f"{k}={v}" for k,v in params.items()])
        with mlflow.start_run(
                nested=True,
                run_name=f"{datetime.date.today().isoformat()}-train-{hparams_suffix}"
        ) as child_run:
            p = mlflow.projects.run(
                uri=str(io_.PROJECT_ROOT.absolute()),
                entry_point=MLFLOW_PROJECT_ENTRYPOINT,
                run_id=child_run.info.run_id,
                parameters=params,

                # The entry point needs the equivalent of the --env-manager=local CLI flag
                #   to use the poetry env of this project
                env_manager="local",

                experiment_id=experiment_id,
                synchronous=fail_on_error,  # If False, the run fails without crashing the current process (script)
            )
            succeeded = p.wait()

            mlflow.log_params(params)

        if succeeded:
            training_run = tracking_client.get_run(p.run_id)
            metrics = training_run.data.metrics

            # cap the loss at the loss of the null model
            # TODO why is null_loss needed?
            # train_loss = min(np.nan, metrics[f"train_loss"])
            # valid_loss = min(null_valid_loss, metrics[f"val_loss"])
            # test_loss = min(null_test_loss, metrics[f"test_loss"])
            train_loss = metrics[f"train_loss"]
            val_loss = metrics[f"val_loss"]
        else:
            # run failed => return null loss
            tracking_client.set_terminated(p.run_id, "FAILED")
            train_loss = np.nan
            val_loss = np.nan

        mlflow.log_metrics(
            {
                f"train_loss": train_loss,
                f"val_loss": val_loss,
            }
        )

        return val_loss

    return eval


@click.command(
    help="Perform hyperparameter search with Hyperopt library. Optimize dl_train target."
)
@click.option("--fail-on-error", type=click.STRING)
@click.option("--algo", type=click.STRING)
@click.option("--max-runs", type=click.INT)
@click.option("--train-batch-size-min", type=click.INT)
@click.option("--train-batch-size-max", type=click.INT)
@click.option("--one-cycle-max-lr-min", type=click.FLOAT)
@click.option("--one-cycle-max-lr-max", type=click.FLOAT)
@click.option("--one-cycle-pct-start-min", type=click.FLOAT)
@click.option("--one-cycle-pct-start-max", type=click.FLOAT)
@click.option("--weight-decay-min", type=click.FLOAT)
@click.option("--weight-decay-max", type=click.FLOAT)
@click.option("--lora-alpha-min", type=click.FLOAT)
@click.option("--lora-alpha-max", type=click.FLOAT)
@click.option("--lora-rank-min", type=click.INT)
@click.option("--lora-rank-max", type=click.INT)
@click.option("--max-epochs", type=click.INT)
@click.option("--accumulate-grad-batches-min", type=click.INT)
@click.option("--accumulate-grad-batches-max", type=click.INT)
@click.option("--limit-batches", type=click.FLOAT)
def train(
        fail_on_error,
        algo,
        max_runs,
        train_batch_size_min,
        train_batch_size_max,
        one_cycle_max_lr_min,
        one_cycle_max_lr_max,
        one_cycle_pct_start_min,
        one_cycle_pct_start_max,
        weight_decay_min,
        weight_decay_max,
        lora_alpha_min,
        lora_alpha_max,
        lora_rank_min,
        lora_rank_max,
        max_epochs,
        accumulate_grad_batches_min,
        accumulate_grad_batches_max,
        limit_batches
):
    # NOTE: Check these references to understand how to use the param space distributions:
    # - https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions
    # - https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html
    #   - here there is a "cheat table" section that suggests which distributions to use
    # - https://www.kaggle.com/code/donkeys/exploring-hyperopt-parameter-tuning
    # NOTE 2: `hp.loguniform(label, low, high)` returns a value drawn according to exp(uniform(low, high)),
    #   meaning that i have to apply the log to the various *min and *max arguments,
    #   because I want the search to be uniform w.r.t. the logarithm of such values (e.g. learning rate)
    space = {
        "train_batch_size": scope.int(hp.quniform(
            "train_batch_size", train_batch_size_min, train_batch_size_max, 1
        )),
        "one_cycle_max_lr": hp.loguniform(
            "one_cycle_max_lr", math.log(one_cycle_max_lr_min), math.log(one_cycle_max_lr_max)
        ),
        "one_cycle_pct_start": hp.uniform(
            "one_cycle_pct_start", one_cycle_pct_start_min, one_cycle_pct_start_max
        ),
        "weight_decay": hp.loguniform(
            "weight_decay", math.log(weight_decay_min), math.log(weight_decay_max)
        ),
        "lora_rank": scope.int(hp.quniform("lora_rank", lora_rank_min, lora_rank_max, 1)),
        "lora_alpha": hp.uniform("lora_alpha", lora_alpha_min, lora_alpha_max),
        "accumulate_grad_batches": scope.int(hp.quniform(  # TODO maybe use choice here in range (min, max)?
            "accumulate_grad_batches", accumulate_grad_batches_min, accumulate_grad_batches_max, 1
        )),
    }

    # mlflow.set_experiment(experiment_name=f"{EXPERIMENT_NAME}")
    mlflow.set_tracking_uri(uri=os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(
        log_system_metrics=True,
        run_name=f"{datetime.date.today().isoformat()}-{MODEL_NAME}",
    ) as run:
        experiment_id = run.info.experiment_id

        # TODO this is not used atm but I could use it to extract additional info
        trials = Trials()

        best = fmin(
            fn=new_eval(
                fail_on_error=fail_on_error != "false",
                experiment_id=experiment_id, 
                max_epochs=max_epochs, 
                limit_batches=limit_batches
            ),
            space=space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
            trials=trials
        )
        mlflow.set_tag("best params", str(best))

        # find the best run, log its metrics as the final metrics of this run
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}' "
        )
        best_val_train = _inf
        best_val_valid = _inf
        best_run = None
        for r in runs:
            if r.data.metrics["val_loss"] < best_val_valid:
                best_run = r
                best_val_train = r.data.metrics["train_loss"]
                best_val_valid = r.data.metrics["val_loss"]
        mlflow.set_tag("best_run", best_run.info.run_id) # TODO I could use run name here maybe?
        mlflow.log_metrics(
            {
                f"train_loss": best_val_train,
                f"val_loss": best_val_valid,
            }
        )


if __name__ == "__main__":
    train()
