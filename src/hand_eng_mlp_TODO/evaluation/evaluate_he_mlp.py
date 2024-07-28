import logging
from datetime import datetime

import datasets
import lightning
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import torch
import torchmetrics.classification as tc
import transformers
from mlflow.entities.model_registry import ModelVersion

import data.common as common
import hand_eng_mlp_TODO.datasets.preprocessing_base as ppb
import hand_eng_mlp_TODO.models.model_beijin as bjn
import training.loader as loader
import utils.mlflow_env as env
from fine_tuned_finbert.datasets.data_modules import Semeval2017Test
from utils.random import RND_SEED


def _main():  # TODO: any implementation about metrics need to be done in Finbert_evaluation as well
    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    model_name = env.get_registered_model_name(loader.Model.HAND_ENG_MLP)
    alias = env.BEST_FULL_TRAINED_MODEL_ALIAS
    client = mlflow.tracking.MlflowClient()
    best_version: ModelVersion = client.get_model_version_by_alias(name=model_name, alias=alias)

    mlflow.set_tag(key='model_name', value=model_name)
    mlflow.set_tag(key='model_alias', value=alias)
    mlflow.set_tag(key='model_version', value=best_version.version)

    model: lightning.LightningModule = mlflow.pytorch.load_checkpoint(
        bjn.ModelBeijin, best_version.run_id,
        kwargs={
            'strict': False,  # Needed because LoRA checkpoint do not include all model parameters
            'log_hparams': True
        }
    )

    def mlflow_evalute_predict(df: pd.DataFrame):
        """
        :param df: pandas df provided by mlflow.evaluate(...)
        :return:
        """
        def collate(tok_output_collection):  # TODO this is for finbert, what should model beijin do?, trasformare dataframe in batch, what am i passing?
            input_ids = torch.stack(list(
                map(
                    lambda x: torch.tensor(x['input_ids'], device=model.device).long(),
                    tok_output_collection
                )
            ))
            att_masks = torch.stack(list(
                map(
                    lambda x: torch.tensor(x['attention_mask'], device=model.device).long(),
                    tok_output_collection
                )
            ))
            tensorized_tokenizer_output = {'input_ids': input_ids, 'attention_mask': att_masks}
            return tensorized_tokenizer_output

        embeddings_col = df[ppb.EMBEDDER_OUTPUT_COL].to_list()  # TODO now embeddings not tokens i need to adapt it
        batches = collate(embeddings_col)

        # Apparently mlflow.evaluate needs cpu tensors or numpy arrays
        return model.predict(**batches).cpu().detach().numpy()


    # TODO ( ͡° ͜ʖ ͡°) create custom mlflow metrics based on what we wrote on the project proposal, so that it gets logged on the mlflow server
    # Create an evaluation function that iterates through the predictions
    # def eval_fn(predictions):
    #     scores = [int(is_valid_python_code(prediction)) for prediction in predictions]
    #     return MetricValue(
    #         scores=scores,
    #         aggregate_results=standard_aggregations(scores),
    #     )

    # Create an EvaluationMetric object for the python code metric
    # valid_code_metric = make_metric(
    #     eval_fn=eval_fn, greater_is_better=False, name="valid_python_code", version="v1"
    # )
    # TODO idk what's this above



    test_dataset: datasets.Dataset = Semeval2017Test().dataset
    pandas_df = test_dataset.to_pandas()

    evaluate_results = mlflow.evaluate(  # TODO see what evaluate do because idk what to add
        model_type='regressor',
        model=mlflow_evalute_predict, #TODO check function above, and see official doc on the browser page i opened
        data=pandas_df,
        feature_names=[ppb.EMBEDDER_OUTPUT_COL],  # TODO now we have embeddings not tokens, i need to adapt it
        targets=common.LABEL_COL,

        # NOTE: this raises the following warning:
        #   WARNING mlflow.models.evaluation.default_evaluator: Skip logging model explainability insights because the shap explainer None requires all feature values to be numeric, and each feature column must only contain scalar values.
        # We are good with this because this kind of model explainability is useless in our case
        evaluators=['default'],

        # TODO add our metrics, e.g. cosine_similarity, f1_score, accuracy, recall, etc.
        extra_metrics=[
        2
        ]
    )




    # TODO EXAMPLE OF EXTRA METRIC USAGE from official documentation
    def root_mean_squared_error(eval_df, _builtin_metrics):
        return np.sqrt((np.abs(eval_df["prediction"] - eval_df["target"]) ** 2).mean)

    rmse_metric = mlflow.models.make_metric(
        eval_fn=root_mean_squared_error,
        greater_is_better=False,
    )
    mlflow.evaluate(..., extra_metrics=[rmse_metric])



    # TODO: sketch from before:
    # TODO: see their cosine similarity - https://alt.qcri.org/semeval2017/task5/index.php?id=evaluation
    class SaharaEvaluator:
        def __init__(self, num_classes: int = 3):
            self._precision: tc.MulticlassPrecision = tc.MulticlassPrecision(num_classes=num_classes)
            self._recall: tc.MulticlassRecall = tc.MulticlassRecall(num_classes=num_classes)
            self._f1: tc.MulticlassF1Score = tc.MulticlassF1Score(num_classes=num_classes)

        # I put + 1 because pytorch recall, precision and f1 require non-negative tensors
        def precision(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self._precision(preds=pred + 1, target=target + 1)

        def recall(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self._recall(preds=pred + 1, target=target + 1)

        def f1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self._f1(preds=pred + 1, target=target + 1)








    # TODO ( ͡° ͜ʖ ͡°) maybe make some plots here with res.metrics and log them
    #   via mlflow.log_artifacts/image/plot whatever the method is
    # evaluate_results.metrics # TODO what's this for? haven't checked yet



    tokenizer = transformers.AutoTokenizer.from_pretrained(
        bjn.PRE_TRAINED_MODEL_PATH, use_fast=True
    )

    def shap_text_predict(texts: np.ndarray):  # TODO: what does this function do?
        tv = tokenizer(
            texts.tolist(),
            padding="max_length",
            max_length=160,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(model.device)

        # IMPORTANT: why to hide manually these special tokens by setting their attention "bit" to 0?
        #   Because shap.plots.text calculates the base_value as the prediction of the model where
        #       all tokens are masked, i.e. something like '[CLS] [MASK] ... [MASK] [SEP]'
        #   It seems, however, that [MASK] tokens (as do all the other ones, even [PAD]), *when attended* by the model,
        #       do have some impact on the output. Since the tokenizer sets the attention_mask to 0 only for
        #       true pad tokens, i.e. padding after the [SEP] (end sentence) token, we have that the base_value
        #       changes depending on the input sentence.
        #   What I would like to do is establish a common, input-length-independent baseline for each sample.
        #   By manually setting attention of [MASK] tokens to 0, we define the baseline
        #       as only the [CLS] and the [SEP] tokens, which intuitively represents the sentiment score
        #       associated to empty inputs of the same length of the current sample.
        #   I tried only using the [CLS] token as baseline, but base_line results made
        #       less sense than in the [CLS]+[SEP] case, although in the latter case base_value
        #       are *slightly* different from each other  (which was not the case with
        #       [CLS]-only since the output was truly constant w.r.t. input length)
        #   The shap values of each token/token-cluster will hence be the difference w.r.t. to an input that
        #       consists of only the [CLS] token and the [SEP] token.
        special_tokens_mask = (tv['input_ids'] == tokenizer.mask_token_id)  # TODO actually now that dropout is fixed try to attend them and see what happens
        # Mask [SEP] too to test what happens, if curious
        # special_tokens_mask = ((tv['input_ids'] == tokenizer.mask_token_id)
        #                           | (tv['input_ids'] == tokenizer.sep_token_id))
        tv['attention_mask'][special_tokens_mask] = 0

        # NOTE: Returning list because only type that I am sure does not cause error`s
        sent_score = model.predict(**tv).detach().cpu().tolist()
        return sent_score



    # TODO what does this do?
    #  from here below (line 206) it's not my domain anymore
    #  and i don't see anything that should be changed depending on the model or dataset
    explainer = shap.Explainer(
        model=shap_text_predict,  # TODO check function above
        masker=tokenizer,
        seed=RND_SEED
    )
    shap_values = explainer(pandas_df['spans'].to_numpy())

    # References to understand SHAP plots:
    # https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
    # https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Using%20custom%20functions%20and%20tokenizers.html#Visualize-the-impact-on-all-the-output-classes
    # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html
    # https://coderzcolumn.com/tutorials/artificial-intelligence/explain-text-classification-models-using-shap-values-keras#6
    # Great book on ML interpretability in general, with a SHAP specific chapter
    #   https://christophm.github.io/interpretable-ml-book/shap.html
    # TLDR: for a our kind of model, it determines the 'base_value' prediction as that of a fully masked text.
    #   It then computes shap values as the expected impact (i.e. the average deviation from the base_value)
    #       that a certain word has on the output score when included, as opposed to it being masked
    html_string = shap.plots.text(
        shap_values,
        display=False,
        xmin=-1,
        xmax=1,
        cmax=1
    )
    mlflow.log_text(html_string, artifact_file="shap-text-plot.html")

    for i in range(len(shap_values)):
        ax = shap.waterfall_plot(shap_values[i, :], max_display=10, show=False)
        fig = ax.figure
        fig.set_size_inches(20, 8)
        mlflow.log_figure(figure=fig, artifact_file=f'shap-waterfall-plots/{i}.png', save_kwargs={'dpi': 100})
        plt.close(fig)

    ax = shap.plots.bar(shap_values, order=shap.Explanation.argsort.flip, max_display=20, show=False)
    fig = ax.figure
    fig.set_size_inches(20, 30)
    mlflow.log_figure(figure=fig, artifact_file=f'shap-barplot-most-important.png', save_kwargs={'dpi': 100})
    plt.close(fig)

    ax = shap.plots.bar(shap_values, order=shap.Explanation.argsort, max_display=20, show=False)
    fig = ax.figure
    fig.set_size_inches(20, 30)
    mlflow.log_figure(figure=fig, artifact_file=f'shap-barplot-least-important.png', save_kwargs={'dpi': 100})
    plt.close(fig)

    for i in range(len(shap_values)):
        ax = shap.force_plot(base_value=shap_values[i], show=False, matplotlib=True)
        fig = ax.figure
        fig.set_size_inches(20, 5)
        fig.tight_layout()
        mlflow.log_figure(figure=ax.figure, artifact_file=f'shap-force-plots/{i}.png', save_kwargs={'dpi': 100})
        plt.close(fig)


if __name__ == '__main__':
    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(env.EVALUATION_EXPERIMENT_NAME_PREFIX)
    with mlflow.start_run(
            log_system_metrics=True,
            run_name=f"{datetime.now().isoformat(timespec='seconds')}-{loader.Model.FINBERT}-evaluation"
    ) as run:
        _main()
