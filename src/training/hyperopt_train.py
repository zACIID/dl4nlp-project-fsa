"""
Reference:
https://github.dev/mlflow/mlflow/blob/master/examples/hyperparam/search_hyperopt.py
"""
import datetime
import math
import os
import typing

import click
import dotenv
import numpy as np
import hyperopt
from hyperopt import fmin, hp, rand, tpe, Trials
from hyperopt.pyll import scope

import mlflow.projects
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

import utils.io as io_
import training.loader as loader

dotenv.load_dotenv(str(io_.PROJECT_ROOT / 'mlflow.env'))

_inf = np.finfo(np.float64).max

EXPERIMENT_NAME_PREFIX = 'Hyperopt'
MLFLOW_TRAIN_ENTRYPOINT = 'train'


def _update_best_metrics(experiment_id, parent_run):
    # find the best run, log its metrics as the final metrics of this run
    client = MlflowClient()
    runs = client.search_runs(
        [experiment_id], f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' "
    )
    best_val_train = _inf
    best_val_valid = _inf
    best_run = None
    for r in runs:
        metric_key = "val_loss_epoch"
        if r.data.metrics.__contains__(metric_key) and r.data.metrics[metric_key] < best_val_valid:
            best_run = r
            best_val_train = r.data.metrics["train_loss_epoch"]
            best_val_valid = r.data.metrics["val_loss_epoch"]

    if best_run is not None:
        mlflow.set_tag("Best Run", best_run.info.run_id)
        metrics = {
            f"train_loss_epoch": best_val_train,
            f"val_loss_epoch": best_val_valid,
        }

        for k, v in metrics.items():
            mlflow.log_metric(
                key=k,
                value=v,
                step=len(runs),
                run_id=parent_run.info.run_id
            )


def new_eval(
        experiment_id,
        parent_run,
        max_epochs,
        limit_batches,
        model_choice,
        dataset_choice,
        with_neutral_samples: str,
        fail_on_error: bool):
    def eval(params: typing.Dict[str, typing.Any]):
        tracking_client = MlflowClient()

        # These are params excluded from hparam search that still
        #   need to be passed to the training run
        params["max_epochs"] = max_epochs
        params["limit_batches"] = limit_batches
        params["model_choice"] = model_choice
        params["dataset_choice"] = dataset_choice
        params["with_neutral_samples"] = with_neutral_samples

        # hparams_suffix = '-'.join([f"{k}={v}" for k,v in params.items()])
        with mlflow.start_run(
                nested=True,
                # run_name=f"{datetime.date.today().isoformat()}-train-{hparams_suffix}" # TODO let's try without hparams because too long
                run_name=f"{datetime.date.today().isoformat()}-train"
        ) as child_run:
            p = mlflow.projects.run(
                uri=str(io_.PROJECT_ROOT.absolute()),
                entry_point=MLFLOW_TRAIN_ENTRYPOINT,
                run_id=child_run.info.run_id,
                parameters=params,

                # The entry point needs the equivalent of the --env-manager=local CLI flag
                #   to use the poetry env of this project
                env_manager="local",
                experiment_id=experiment_id,
                synchronous=fail_on_error,  # If False, the run fails without crashing the current process (script)
            )
            succeeded = p.wait()

        if succeeded:
            training_run = tracking_client.get_run(p.run_id)
            metrics = training_run.data.metrics

            val_loss = metrics[f"val_loss_epoch"]
            status = hyperopt.STATUS_OK
        else:
            # run failed => return null loss
            tracking_client.set_terminated(run_id=p.run_id, status=RunStatus.to_string(RunStatus.FAILED))
            val_loss = np.nan
            status = hyperopt.STATUS_FAIL

        _update_best_metrics(experiment_id, parent_run)

        # Check here to see what should be returned:
        # https://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/#attaching-extra-information-via-the-trials-object
        return {
            'status': status,
            'loss': val_loss
        }

    return eval


@click.command(
    help="Perform hyperparameter search with Hyperopt library"
)
@click.option("--model-choice", default=loader.Model.FINBERT.value, type=click.STRING)
@click.option("--dataset-choice", default=loader.Dataset.SC_TRAIN_SEMEVAL_VAL.value, type=click.STRING)
@click.option("--with-neutral-samples", default="false", type=click.STRING)
@click.option("--algo", default='tpe.suggest', type=click.STRING)
@click.option("--max-runs", default=10, type=click.INT)
@click.option("--one-cycle-max-lr-min", default=1e-5, type=click.FLOAT)
@click.option("--one-cycle-max-lr-max", default=5e-3, type=click.FLOAT)
@click.option("--one-cycle-pct-start-min", default=0.05, type=click.FLOAT)
@click.option("--one-cycle-pct-start-max", default=0.5, type=click.FLOAT)
@click.option("--weight-decay-min", default=5e-5, type=click.FLOAT)
@click.option("--weight-decay-max", default=1e-2, type=click.FLOAT)
@click.option("--lora-alpha-min", default=0.5, type=click.FLOAT)
@click.option("--lora-alpha-max", default=3, type=click.FLOAT)
@click.option("--lora-rank-min", default=64, type=click.INT)
@click.option("--lora-rank-max", default=256, type=click.INT)
@click.option("--max-epochs", default=10, type=click.INT)
@click.option("--accumulate-grad-batches-min", default=4, type=click.INT)
@click.option("--accumulate-grad-batches-max", default=15, type=click.INT)
@click.option("--limit-batches", default=0.001, type=click.FLOAT)
@click.option("--fail-on-error", default='true', type=click.STRING)
def train(
        model_choice,
        dataset_choice,
        with_neutral_samples,
        algo,
        max_runs,
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
        limit_batches,
        fail_on_error,
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
        "accumulate_grad_batches": scope.int(hp.quniform(
            "accumulate_grad_batches", accumulate_grad_batches_min, accumulate_grad_batches_max, 1
        )),
    }

    mlflow.set_tracking_uri(uri=os.environ["MLFLOW_TRACKING_URI"])
    # experiment = mlflow.set_experiment(experiment_name=f"{EXPERIMENT_NAME_PREFIX} | {dataset_choice} | {model_choice}")
    with mlflow.start_run(
        log_system_metrics=True,
        run_name=f"{datetime.date.today().isoformat()}",
    ) as run:
        experiment_id = run.info.experiment_id

        # TODO this is not used atm but I could use it to extract additional info
        trials = Trials()

        best = fmin(
            fn=new_eval(
                fail_on_error=fail_on_error != "false",
                experiment_id=experiment_id,
                parent_run=run,
                max_epochs=max_epochs, 
                limit_batches=limit_batches,
                model_choice=model_choice,
                dataset_choice=dataset_choice,
                with_neutral_samples=with_neutral_samples
            ),
            space=space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
            trials=trials
        )
        mlflow.set_tag("Best Params", str(best))


if __name__ == "__main__":
    train()
