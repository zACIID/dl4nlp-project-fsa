"""
Reference:
https://github.dev/mlflow/mlflow/blob/master/examples/hyperparam/search_hyperopt.py
"""
import datetime
import math
import os
import typing

import click
import numpy as np
import hyperopt
from hyperopt import fmin, hp, rand, tpe, Trials
from hyperopt.pyll import scope

import mlflow.projects
from mlflow import ActiveRun
from mlflow.entities import RunStatus, Experiment
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

import utils.io as io_
import utils.mlflow_env as env
from training.loader import Model

_inf = np.finfo(np.float64).max

MLFLOW_TRAIN_ENTRYPOINT = 'train'

VAL_METRIC_KEY = 'val_loss'
TRAIN_METRIC_KEY = 'train_loss'


def _update_best_model(experiment: Experiment, eval_run: ActiveRun):
    # find the best run, log its metrics as the final metrics of this run
    client = MlflowClient()

    model_name = env.get_registered_model_name(env.get_model_choice())
    results = client.search_registered_models(filter_string=f"name = '{model_name}'")
    registered_model = client.get_registered_model(model_name) if len(results) > 0 else None

    if registered_model is not None:
        current_best_model: ModelVersion = registered_model.latest_versions[-1]
        current_best_run = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            # Filter string syntax reference:
            # https://mlflow.org/docs/latest/search-runs.html
            filter_string=f"attributes.run_id = '{current_best_model.run_id}'"
        )[0]

        best_val_train = current_best_run.data.metrics[TRAIN_METRIC_KEY]
        best_val_valid = current_best_run.data.metrics[VAL_METRIC_KEY]
    else:
        best_val_train = _inf
        best_val_valid = _inf

    runs = client.search_runs(
        [experiment.experiment_id], f"attributes.run_id = '{eval_run.info.run_id}' "
    )
    best_run = None
    for r in runs:
        if r.data.metrics.__contains__(VAL_METRIC_KEY) and r.data.metrics[VAL_METRIC_KEY] < best_val_valid:
            best_run = r
            best_val_train = r.data.metrics[TRAIN_METRIC_KEY]
            best_val_valid = r.data.metrics[VAL_METRIC_KEY]

    if best_run is not None:
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics({
            TRAIN_METRIC_KEY: best_val_train,
            VAL_METRIC_KEY: best_val_valid,
        })

        # Add (overwrite) best model to registry
        version = mlflow.register_model(
            model_uri=f"runs:/{best_run.info.run_id}/artifacts/model",
            name=model_name,
            tags=env.get_run_tags()
        )
        client.set_registered_model_alias(
            name=model_name,
            alias=env.BEST_TUNED_MODEL_ALIAS,
            version=version.version
        )
        client.set_registered_model_alias(
            name=model_name,
            alias=env.get_dataset_specific_best_model_alias(dataset=env.get_dataset_choice(), tuning=True),
            version=version.version
        )


def new_eval(
        experiment: Experiment,
        max_epochs,
        limit_batches,
        with_neutral_samples: str,
        fail_on_error: bool):
    def eval(params: typing.Dict[str, typing.Any]):
        tracking_client = MlflowClient()

        # These are params excluded from hparam search that still
        #   need to be passed to the training run
        params["max_epochs"] = max_epochs
        params["limit_batches"] = limit_batches
        params["with_neutral_samples"] = with_neutral_samples

        with mlflow.start_run(
            nested=True,
            run_name=f"{datetime.date.today().isoformat()}-train",
            tags=env.get_run_tags()
        ) as eval_run:
            p = mlflow.projects.run(
                uri=str(io_.PROJECT_ROOT.absolute()),
                entry_point=MLFLOW_TRAIN_ENTRYPOINT,
                run_id=eval_run.info.run_id,
                parameters=params,

                # The entry point needs the equivalent of the --env-manager=local CLI flag
                #   to use the poetry env of this project
                env_manager="local",
                experiment_id=experiment.experiment_id,
                synchronous=fail_on_error,  # If False, the run fails without crashing the current process (script)
            )
            succeeded = p.wait()

        if succeeded:
            training_run = tracking_client.get_run(p.run_id)
            metrics = training_run.data.metrics

            val_loss = metrics[VAL_METRIC_KEY]
            status = hyperopt.STATUS_OK
        else:
            # run failed => return null loss
            tracking_client.set_terminated(run_id=p.run_id, status=RunStatus.to_string(RunStatus.FAILED))
            val_loss = np.nan
            status = hyperopt.STATUS_FAIL

        _update_best_model(experiment, eval_run=eval_run)

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
@click.option("--lora-dropout-min", default=0.1, type=click.FLOAT)
@click.option("--lora-dropout-max", default=0.6, type=click.FLOAT)
@click.option("--lora-rank-min", default=8, type=click.INT)
@click.option("--lora-rank-max", default=256, type=click.INT)
@click.option("--max-epochs", default=10, type=click.INT)
@click.option("--accumulate-grad-batches-min", default=4, type=click.INT)
@click.option("--accumulate-grad-batches-max", default=15, type=click.INT)
@click.option("--limit-batches", default=0.001, type=click.FLOAT)
@click.option("--fail-on-error", default='true', type=click.STRING)
def tune(
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
        lora_dropout_min,
        lora_dropout_max,
        lora_rank_min,
        lora_rank_max,
        max_epochs,
        accumulate_grad_batches_min,
        accumulate_grad_batches_max,
        limit_batches,
        fail_on_error,
):
    env.set_common_run_tags(with_neutral_samples=with_neutral_samples)

    # NOTE: Check these references to understand how to use the param space distributions:
    # - https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions
    # - https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html
    #   - here there is a "cheat table" section that suggests which distributions to use
    # - https://www.kaggle.com/code/donkeys/exploring-hyperopt-parameter-tuning
    # NOTE 2: `hp.loguniform(label, low, high)` returns a value drawn according to exp(uniform(low, high)),
    #   meaning that i have to apply the log to the various *min and *max arguments,
    #   because I want the search to be uniform w.r.t. the logarithm of such values (e.g. learning rate)
    if env.get_model_choice() == Model.FINBERT:
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
            "lora_dropout": hp.uniform("lora_dropout", lora_dropout_min, lora_dropout_max),
            "accumulate_grad_batches": scope.int(hp.quniform(
                "accumulate_grad_batches", accumulate_grad_batches_min, accumulate_grad_batches_max, 1
            )),
        }
    elif env.get_model_choice() == Model.HAND_ENG_MLP:
        space = None
        # TODO ( ͡° ͜ʖ ͡°) implement
        #   How to add parameters to the tuning and training scripts:
        #   1. specify them as click.option in each script (hyperopt_tuning.py, train.py)
        #   2. specify them as params in the MLproject file and pass them inside the "command" section of each entry point
        #   3. profit
        #   It might become a mess because of too many parameters but it is the easy way for now I think
        #   A refactoring somehow for example to split training scripts or at least bring out somewhere
        #       else those sections that depend on a specific model would not be bad, for sure
        raise NotImplementedError('TODO implement hparam search')
    else:
        raise NotImplementedError('Unhandled model choice')

    mlflow.set_tracking_uri(uri=os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(
        log_system_metrics=True,
        run_name=f"{datetime.date.today().isoformat()}",
        tags=env.get_run_tags()
    ) as run:
        experiment = mlflow.get_experiment(run.info.experiment_id)

        # TODO this is not used atm but I could use it to extract additional info
        trials = Trials()

        best = fmin(
            fn=new_eval(
                fail_on_error=fail_on_error != "false",
                experiment=experiment,
                max_epochs=max_epochs,
                limit_batches=limit_batches,
                with_neutral_samples=with_neutral_samples
            ),
            space=space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
            trials=trials
        )
        mlflow.set_tag("best_params", str(best))


if __name__ == "__main__":
    tune()
