# MLflow Stuff

## Create and Connect to the Tracking Server

Run `mlflow server --host 127.0.0.1 --port 5000` (or `mlflow ui`? what's the difference?) (use host 0.0.0.0 if inside Docker container).
By setting the env variable `MLFLOW_TRACKING_URI=http://<host>:<port>`, I do not need to call `mlflow.set_tracking_uri()` from code.

Check the docs:
- [MLflow Tracking Server Quickstart](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html#method-1-start-your-own-mlflow-server)
- [MLflow Tracking Server Docs](https://mlflow.org/docs/latest/tracking/server.html)

### Setting the Experiment

Various way to set the experiment:
- call [mlflow.set_experiment(name)](https://mlflow.org/docs/latest/tracking/server.html), which gets or creates an experiment with the given name
- set the env variables `MLFLOW_EXPERIMENT_NAME` or `MLFLOW_EXPERIMENT_NAME`
- pass the `--experiment_id` arg to `mlflow run`

### About Runs

Usually created inside the code via [mlflow.start_run](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run).
Argument `nested=true` makes runs nested under a parent run, I think useful so that the parent run represents a script launch, and the children runs represent each individual model run inside the script.


## How to log? TODO

#todo check how to log via pytorchlightning and in general. Need to log:
- models
- parameters

#todo check how to retrieve best models from model registry and see if they are compatible out of the box with pytorch lightning


## MLflow Projects

Example partially taken from [Unleashing the power of MLflow](https://freedium.cfd/https://towardsdatascience.com/unleashing-the-power-of-mlflow-36c17a693033):

```yaml
name: My_Project
docker_env:
  image: mlflow-docker-example-environment
  volumes: ["/local/path:/container/mount/path"]
  environment: [["NEW_ENV_VAR", "new_var_value"], "VAR_TO_COPY_FROM_HOST_ENVIRONMENT"]
entry_points:
  main:
    parameters:
      alpha: float # no default value
      l1_ratio: {type: float, default: 0.5} # default value
    command: "python main.py {alpha} {l1_ratio}" # run script with params
  secondary:
    parameters:
      param1: {type: float, default: 0}
    command: "python secondary.py {param1}" # run script with params
```

To launch one of the scripts (entry points), the `-e` flag must be used:

```bash
mlflow run . -e main -P alpha=0.6
```

The project is automatically run inside the docker environment.

Note that the first argument `.` is the project uri and denotes that the project is to be run in the current directory. [See here](https://mlflow.org/docs/latest/projects.html#running-projects) for other project uri options and, in general, for more details on the `mlflow run` CLI

### How to use Poetry and MLflow?

[Poetry isn't supported natively](https://github.com/mlflow/mlflow/issues/9717) by MLflow as an env manager. There are two ways to combine it with MLflow, apparently:
- use poetry inside a Docker image and then run the project with the docker environment
- use `--env-manager local` (check [the docs](https://mlflow.org/docs/latest/cli.html#mlflow-run)) and then possibly in the command section of the entry points prepend `<poetry env activate (whatever the command)> && ...`


## Handling Experiment Scripts Arguments

### ArgumentParser

Classic handling by using the ArgumentParser class. See [this example](https://github.com/chauhang/mlflow/blob/master/examples/pytorch/MNIST/example1/mnist_autolog_example1.py) to understand how it was used to both add custom arguments and automatically add the parameters specified by for example PyTorch Lightning's `Trainer` class via `Trainer.add_argparse_args(parser=parser)`

### MLflow + Hydra

Interesting template that uses a library called [hydra](https://hydra.cc/docs/intro/#basic-example) to handle configurations: [mlflow + hydra + poetry template](https://github.com/hppRC/template-pytorch-lightning-hydra-mlflow-poetry).

NOTE: in the example's `config.yaml`, the property `defaults` specifies which configuration file of the respective folders (i.e. the sub-property names) to extend `config.yaml` with.

I think Hydra can be useful in my case because I might want to specify all the training script args into a .yaml config file, load it with Hydra and then pass it down to the script.
I'd still like to be able to override stuff from command line because it is faster? But maybe I am confined to yaml config files. At least I can specify via CLI what is the config file that I want to use.

*UPDATE*: It seems that I can overide config values via CLI arguments, as shown in the `# Template Design` section of the above mentioned hydra + poetry + mlflow template

### Lightning CLI

**I think this is it**: this basically provides the same functionality that I desired Hydra for, that is being able to define yaml config files.
It provides automatic instantiation of objects and does A LOT of stuff under the hood.
See the docs:
- [Main Reference](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)

Check the `--print-config` option to have the yaml file with every available option, even model/datamodule specific, so that configuration is easier

**UPDATE: THIS MAY NOT BE IT** because I do not know If I can provide a custom body to the launch script that includes the Lightning CLI, which I need because I have to wrap stuff with mlflow and then use hyperopt to search params.
If this is the unfortunate case, then I have to resort to hydra and manually setup the config file
**UPDATE2: IT MAY BE PARTIALLY IT**: 
- I think this is useful for scripts that do not involve changing parameters, because parameter tuning algos must be able to pass stuff directly to the classes called
- I can still use LightningCLI inside a script and call it via an MLflow project entry point, so that via code I can use `mlflow.projects.run(...)` and invoke the entry points with the parameters that I need

## Other Stuff

### What I'd like to do

I think Docker + Hydra + MLflow project will be the way I'll run the project.

### Limitations of Self-hosted MLflow

[check this article](https://neptune.ai/blog/best-mlflow-alternatives#:~:text=MLflow%20is%20a%20popular%20open,learning%20lifecycle%20and%20facilitate%20reproducibility.)

### Useful MLflow Examples

#### Example of Pipeline with MLflow (Multistep Workflow)

Read this short [MLflow Docs section](https://github.dev/mlflow/mlflow/blob/master/examples/hyperparam/search_hyperopt.py) and then check this [Example | GitHub](https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/README.rst)

#### Example of Hyperaparms Training with MLflow

[Example | GitHub](https://github.dev/mlflow/mlflow/blob/master/examples/hyperparam/search_hyperopt.py). In particular, check search_hyperopt.py and try to augment it with Hydra params: such a script has fixed ranges for param search, but I'd also like to parameterize such ranges.
