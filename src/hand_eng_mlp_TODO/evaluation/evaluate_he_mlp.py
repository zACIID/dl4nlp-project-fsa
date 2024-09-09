import logging
import typing
from datetime import datetime
from typing import List

import datasets
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import torch
import transformers
from mlflow.metrics import MetricValue
from mlflow.models.evaluation import make_metric
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

import data.common as common
import data.stocktwits_crypto_dataset as sc
import hand_eng_mlp_TODO.datasets.preprocessing_base as ppb
import hand_eng_mlp_TODO.datasets.preprocessing_features_extraction as ppfe
import hand_eng_mlp_TODO.models.model_beijin as hemlp
import training.loader as loader
import utils.io as io_
import utils.mlflow_env as env
from hand_eng_mlp_TODO.datasets.data_modules import Semeval2017Test
from utils.random import RND_SEED


# TODO:
#   - refactor, even in finbert, so that functions to create metrics and shap_text_predict embeddings can be imported
#   - fix finbert too
#   - implement evaluate for e2e model

# Limiting the number of samples provided to the SHAP explainer to keep computation times low
SHAP_EXPLAINER_MAX_SAMPLES = 25


def _main():  # TODO: any implementation about metrics need to be done in Finbert_evaluation as well
    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    model: hemlp.ModelBeijin = loader.load_best_model(loader.Model.HAND_ENG_MLP).cpu()
    model.eval()

    # Our metrics:
    # Main metric: cosine similarity, the SemEval2017 challenge's official evaluation method.
    # SemEval2017 cosine similarity - https://alt.qcri.org/semeval2017/task5/index.php?id=evaluation
    # Defined as: cosine(G,P)= \frac{sum_{i=0}^{n} G_i x P_i}{\sqrt{sum_{i=0}^{n} G_i^2} x \sqrt{sum_{i=0}^{n} P_i^2}}
    # Additional standard metrics, including precision, recall, and F1 score, will be considered.

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

    test_df: pd.DataFrame = Semeval2017Test().dataset.to_pandas()

    # Remove embedder col and pass it "from outside" because it can't be serialized by mlflow.evaluate(),
    #   being a column of numpy arrays
    embedder_col: pd.Series = test_df[ppb.EMBEDDER_OUTPUT_COL]
    test_df = test_df.drop(columns=[ppb.EMBEDDER_OUTPUT_COL])

    def mlflow_evaluate_predict(df: pd.DataFrame): #TODO
        """
        :param df: pandas df provided by mlflow.evaluate(...)
        :return:
        """

        def _collate_fn(embedder_col: pd.Series, new_features_cols: typing.List[pd.Series]):
            embeddings = [torch.tensor(embedding.tolist()) for embedding in embedder_col]
            embeddings_batch = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

            new_features = [torch.tensor(feats.values) for feats in new_features_cols]
            new_features_batch = torch.stack(new_features).T

            return embeddings_batch, new_features_batch

        feature_col = [df[col] for col in ppfe.NEW_FEATURES]
        embeddings, features = _collate_fn(embedder_col, feature_col)

        # Apparently mlflow.evaluate needs cpu tensors or numpy arrays
        return model.predict(embeddings, features).cpu().detach().numpy()

    evaluate_results = mlflow.evaluate(
        model_type='regressor',
        model=mlflow_evaluate_predict,
        data=test_df,
        feature_names=[*ppfe.NEW_FEATURES],
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

    def shap_text_predict(texts: np.ndarray):
        embeddings: typing.List[torch.Tensor] = _apply_tokenize_and_embed(texts)
        features = np.array([list(_extract_features(text).values()) for text in texts])
        np.nan_to_num(features, nan=0)

        embeddings_tensor = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        features_tensor = torch.tensor(features).to(dtype=torch.float32)

        with torch.no_grad():
            predictions = model.predict(x_batch=embeddings_tensor, beijin_feats_batch=features_tensor)

        return predictions.cpu().tolist()

    tokenizer = AutoTokenizer.from_pretrained(hemlp.PRE_TRAINED_MODEL_PATH, use_fast=True)
    bertweet = AutoModel.from_pretrained(hemlp.PRE_TRAINED_MODEL_PATH)
    def _apply_tokenize_and_embed(texts: np.ndarray) -> List[torch.Tensor]:
        bertweet.eval()

        def tokenize_and_embed(text: str) -> torch.Tensor:
            inputs = tokenizer(
                text,
                padding=False,
                return_tensors="pt",
                truncation=True,
                max_length=sc.WORST_CASE_TOKENS
            )
            with torch.no_grad():
                outputs = bertweet(**inputs)  # **inputs unpacks the dictionary returned by the tokenizer

            embeddings = outputs.last_hidden_state[:, 1:-1, :].squeeze(dim=0)
            return embeddings

        return [tokenize_and_embed(text) for text in texts]

    sentiment_data, default_mean_value = ppfe.load_sentiment_dataset(io_.DATA_DIR)
    def _extract_features(text: str) -> typing.Dict[str, float]:
        features = {}
        features['vader_polarity'] = ppfe.compute_vader_polarity(text)

        vader_features = ppfe.calculate_vader_pos_neg_features(text)
        features['pos_neg_ratio_vader'] = vader_features.pos_neg_ratio
        features['pos_neg_difference_vader'] = vader_features.pos_neg_difference
        features['sentiment_entropy_vader'] = ppfe.calculate_vader_sentiment_entropy(text)

        features['swn_polarity'] = ppfe.compute_swn_polarity(text)

        sentic_emotion = ppfe.sentic_emotion_recognition(text)
        features["INTROSPECTION"] = sentic_emotion["INTROSPECTION"]
        features["TEMPER"] = sentic_emotion["TEMPER"]
        features["ATTITUDE"] = sentic_emotion["ATTITUDE"]
        features["SENSITIVITY"] = sentic_emotion["SENSITIVITY"]

        readability_metrics = ppfe.calculate_readability_metrics(text)
        features.update({
            'flesch_kincaid_grade': readability_metrics.flesch_kincaid_grade,
            'gunning_fog': readability_metrics.gunning_fog,
            'coleman_liau_index': readability_metrics.coleman_liau_index
        })

        sentiment_features = ppfe.compute_overall_sentiment_features(text, sentiment_data, default_mean_value)
        features.update({
            'overall_valence_mean': sentiment_features.overall_valence_mean,
            'overall_arousal_mean': sentiment_features.overall_arousal_mean,
            'overall_dominance_mean': sentiment_features.overall_dominance_mean,
            'overall_valence_std': sentiment_features.overall_valence_std,
            'overall_arousal_std': sentiment_features.overall_arousal_std,
            'overall_dominance_std': sentiment_features.overall_dominance_std,
            'valence_contrast': sentiment_features.valence_contrast,
            'arousal_contrast': sentiment_features.arousal_contrast,
            'dominance_contrast': sentiment_features.dominance_contrast
        })

        return features

    explainer = shap.Explainer(
        model=shap_text_predict,
        masker=tokenizer,
        seed=RND_SEED
    )
    shap_values = explainer(test_df['spans'][:SHAP_EXPLAINER_MAX_SAMPLES].to_numpy())

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
