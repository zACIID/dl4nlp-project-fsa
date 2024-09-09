import logging
from datetime import datetime

import datasets
import lightning
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
import pandas as pd
import shap
import torch
import transformers
from mlflow.entities.model_registry import ModelVersion
from mlflow.metrics import MetricValue
from mlflow.models.evaluation import make_metric
from sklearn.metrics import precision_score, recall_score, f1_score

import data.common as common
import fine_tuned_finbert.datasets.preprocessing_base as ppb
import fine_tuned_finbert.models.fine_tuned_finbert as ft
import training.loader as loader
import utils.mlflow_env as env
from fine_tuned_finbert.datasets.data_modules import Semeval2017Test
from utils.random import RND_SEED


def _main():
    # shap.initjs() Needed only in notebook environment

    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    model: ft.FineTunedFinBERT = loader.load_best_model(loader.Model.FINBERT).cpu()
    model.eval()

    def mlflow_evalute_predict(df: pd.DataFrame):
        """
        :param df: pandas df provided by mlflow.evaluate(...)
        :return:
        """
        def collate(tokenizer_col):
            input_ids = torch.stack(list(
                map(
                    lambda x: torch.tensor(x['input_ids'], device=model.device).long(),
                    tokenizer_col
                )
            ))
            att_masks = torch.stack(list(
                map(
                    lambda x: torch.tensor(x['attention_mask'], device=model.device).long(),
                    tokenizer_col
                )
            ))
            tensorized_tokenizer_output = {'input_ids': input_ids, 'attention_mask': att_masks}
            return tensorized_tokenizer_output

        tokenizer_col = df[ppb.TOKENIZER_OUTPUT_COL].to_list()
        batches = collate(tokenizer_col)

        # Apparently mlflow.evaluate needs cpu tensors or numpy arrays
        return model.predict(**batches).cpu().detach().numpy()

    # Thresholding predictions and targets: [-1,-0.25)=negative, [-0.25,0.25]=neutral, (0.25,1]=positive
    def apply_thresholds(values):
        return np.where(values < -0.25, -1, np.where(values > 0.25, 1, 0))

    def cosine_similarity(y_true, y_pred):
        cos_sim = np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
        return cos_sim

    # Evaluation functions that compute Cosine similarity, Precision, Recall, F1 score
    def eval_fn_cosine_similarity(predictions, targets):
        score = cosine_similarity(predictions, targets)
        return MetricValue(aggregate_results={"cosine_similarity": score})

    def eval_fn_precision(predictions, targets):
        predictions = apply_thresholds(predictions)
        targets = apply_thresholds(targets)
        score = precision_score(targets, predictions, average='weighted')
        return MetricValue(aggregate_results={"precision": score})

    def eval_fn_recall(predictions, targets):
        predictions = apply_thresholds(predictions)
        targets = apply_thresholds(targets)
        score = recall_score(targets, predictions, average='weighted')
        return MetricValue(aggregate_results={"recall": score})

    def eval_fn_f1(predictions, targets):
        predictions = apply_thresholds(predictions)
        targets = apply_thresholds(targets)
        score = f1_score(targets, predictions, average='weighted')
        return MetricValue(aggregate_results={"f1": score})

    # Create EvaluationMetric for all metrics
    cosine_similarity_metric = make_metric(eval_fn=eval_fn_cosine_similarity, greater_is_better=True,
                                           name="cosine_similarity", version="v1")
    precision_metric = make_metric(eval_fn=eval_fn_precision, greater_is_better=True, name="precision", version="v1")
    recall_metric = make_metric(eval_fn=eval_fn_recall, greater_is_better=True, name="recall", version="v1")
    f1_metric = make_metric(eval_fn=eval_fn_f1, greater_is_better=True, name="f1_score", version="v1")

    test_dataset: datasets.Dataset = Semeval2017Test().dataset
    pandas_df = test_dataset.to_pandas()

    evaluate_results = mlflow.evaluate(
        model_type='regressor',
        model=mlflow_evalute_predict,
        data=pandas_df,
        feature_names=[ppb.TOKENIZER_OUTPUT_COL],
        targets=common.LABEL_COL,

        # NOTE: this raises the following warning:
        #   WARNING mlflow.models.evaluation.default_evaluator: Skip logging model explainability insights because the shap explainer None requires all feature values to be numeric, and each feature column must only contain scalar values.
        # We are good with this because this kind of model explainability is useless in our case
        evaluators=['default'],

        extra_metrics=[
            cosine_similarity_metric,
            precision_metric,
            recall_metric,
            f1_metric
        ]
    )

    # TODO ( ͡° ͜ʖ ͡°) maybe make some plots here with res.metrics and log them
    #   via mlflow.log_artifacts/image/plot whatever the method is
    metrics = evaluate_results.metrics
    metrics_dict = {
        "Cosine Similarity": metrics["cosine_similarity"].score,
        "Precision": metrics["precision"].score,
        "Recall": metrics["recall"].score,
        "F1 Score": metrics["f1_score"].score,
    }

    # for metric_name, metric_value in metrics_dict.items():
    #     plt.figure(figsize=(6, 4))
    #     sns.barplot(x=[metric_name], y=[metric_value])
    #     plt.title(f"{metric_name} Value")
    #     plt.xlabel("Metric")
    #     plt.ylabel("Value")
    #     plt.ylim(0, 1)
    #     plt.tight_layout()
    #
    #     plot_filename = f"{metric_name.lower().replace(' ', '_')}_barplot.png"
    #     plt.savefig(plot_filename)
    #     mlflow.log_artifact(plot_filename)
    #     plt.close()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ft.PRE_TRAINED_MODEL_PATH, use_fast=True
    )

    def shap_text_predict(texts: np.ndarray):
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
        special_tokens_mask = (tv['input_ids'] == tokenizer.mask_token_id)
        # Mask [SEP] too to test what happens, if curious
        # special_tokens_mask = ((tv['input_ids'] == tokenizer.mask_token_id)
        #                           | (tv['input_ids'] == tokenizer.sep_token_id))
        tv['attention_mask'][special_tokens_mask] = 0

        # NOTE: Returning list because only type that I am sure does not cause error`s
        sent_score = model.predict(**tv).detach().cpu().tolist()
        return sent_score

    explainer = shap.Explainer(
        model=shap_text_predict,
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
