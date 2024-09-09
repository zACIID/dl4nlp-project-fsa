import logging
from datetime import datetime

import datasets
import mlflow
import pandas as pd
from mlflow.models.evaluation import make_metric

import data.common as common
import end_to_end_TODO.data_modules as dm
import end_to_end_TODO.models.end_to_end_model as e2e
import fine_tuned_finbert.datasets.preprocessing_base as ppb_ft
import fine_tuned_finbert.evaluation.utils as eu_ft
import hand_eng_mlp_TODO.datasets.preprocessing_base as ppb_hemlp
import hand_eng_mlp_TODO.datasets.preprocessing_features_extraction as ppfe
import hand_eng_mlp_TODO.evaluation.utils as eu_hemlp
import training.loader as loader
import utils.evaluation as eval_utils
import utils.mlflow_env as env

# Limiting the number of samples provided to the SHAP explainer to keep computation times low
SHAP_EXPLAINER_MAX_SAMPLES = 25


def _main():  # TODO: any implementation about metrics need to be done in Finbert_evaluation as well
    pytorch_logger = logging.getLogger("lightning.pytorch")
    pytorch_logger.setLevel(logging.INFO)

    model: e2e.EndToEndModel = loader.load_best_model(loader.Model.END_TO_END).cpu()
    model.eval()

    # Create EvaluationMetric for all metrics
    cosine_similarity_metric = make_metric(eval_fn=eval_utils.eval_fn_cosine_similarity, greater_is_better=True,
                                           name="cosine_similarity", version="v1")
    precision_metric = make_metric(eval_fn=eval_utils.eval_fn_precision, greater_is_better=True, name="precision", version="v1")
    recall_metric = make_metric(eval_fn=eval_utils.eval_fn_recall, greater_is_better=True, name="recall", version="v1")
    f1_metric = make_metric(eval_fn=eval_utils.eval_fn_f1, greater_is_better=True, name="f1_score", version="v1")

    test_dataset: datasets.Dataset = dm.Semeval2017Test().dataset
    test_df = test_dataset.to_pandas()

    # Remove embedder col and pass it "from outside" because it can't be serialized by mlflow.evaluate(),
    #   being a column of numpy arrays
    embedder_col: pd.Series = test_df[ppb_hemlp.EMBEDDER_OUTPUT_COL]
    test_df = test_df.drop(columns=[ppb_hemlp.EMBEDDER_OUTPUT_COL])

    def mlflow_evaluate_predict(df: pd.DataFrame): #TODO
        """
        :param df: pandas df provided by mlflow.evaluate(...)
        :return:
        """
        feature_col = [df[col] for col in ppfe.NEW_FEATURES]
        embeddings, features = eu_hemlp.collate(embedder_col, feature_col, model)
        tokenizer_col = df[ppb_ft.TOKENIZER_OUTPUT_COL].to_list()
        ft_batches = eu_ft.collate(tokenizer_col, model)

        # Apparently mlflow.evaluate needs cpu tensors or numpy arrays
        return model.predict((ft_batches, embeddings, features)).cpu().detach().numpy()

    evaluate_results = mlflow.evaluate(
        model_type='regressor',
        model=mlflow_evaluate_predict,
        data=test_df,
        feature_names=[*ppfe.NEW_FEATURES, ppb_ft.TOKENIZER_OUTPUT_COL],
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


    # TODO currently impractical to compute SHAP values because of two different tokenizers and
    #   huge time required for hemlp model -> if/when implemented,
    #   continue to refactor the two evalute_finbert and evaluate_hemlp so that extracted functions can be used here
    #   the problem of mlp stuff being slow is probably related to the warning "from numpy to tensor it's extremely slow"


if __name__ == '__main__':
    mlflow.set_tracking_uri(env.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(env.EVALUATION_EXPERIMENT_NAME_PREFIX)
    with mlflow.start_run(
            log_system_metrics=True,
            run_name=f"{datetime.now().isoformat(timespec='seconds')}-{loader.Model.HAND_ENG_MLP}-evaluation"
    ) as run:
        _main()
