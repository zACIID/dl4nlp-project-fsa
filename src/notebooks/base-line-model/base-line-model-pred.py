# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + pycharm={"name": "#%%\n", "is_executing": true}
import os

from transformers import RobertaForMaskedLM

from hand_eng_mlp_TODO.models.linear_aggregator import LinAggregator
from src.fine_tuned_finbert.datasets.data_modules.semeval_2017_test import Semeval2017Test
import src.hand_eng_mlp_TODO.datasets.data_modules as dm_hemlp
from src.baseline_model_TODO.model.base_line_but_still_sauced import BLBSSModel
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import hand_eng_mlp_TODO.models.model_beijin as hemlp
from src.utils.io import PROJECT_ROOT
from src.baseline_model_TODO.model.base_line_but_still_sauced import BLBSSType
import lightning as L
from typing import Type

# + pycharm={"name": "#%%\n"}

# -

# ## FinBERT baseline model

# ### Set up dataset

# + pycharm={"name": "#%%\n"}
data_handler = Semeval2017Test()
data_handler.setup()

data_loader: DataLoader = data_handler.test_dataloader()

fin_bert_bl_model: BLBSSModel = BLBSSModel(model_type=BLBSSType.FinBERT)


# -

# ### Testing

# + pycharm={"name": "#%%\n"}
def fin_bert_predict(model: nn.Module, data_loader: DataLoader, path: str) -> None:
    predictions: np.ndarray[float] = np.array([], dtype=np.float_)
    model.eval()

    with torch.no_grad():
        for batch, _ in data_loader:
            batch_pred: np.ndarray[float] = model.predict(finbert_input=batch).cpu().numpy()
            predictions = np.concatenate((predictions, batch_pred))


    df: pd.DataFrame = pd.DataFrame(predictions)
    df.to_csv(path, index=False)
    
# path = PROJECT_ROOT / "data" / "evaluation" / "base_model_pred_fin_bert.csv"
# 
# fin_bert_predict(model=fin_bert_bl_model, data_loader=data_loader, path=path)

# + pycharm={"name": "#%%\n"}
ft_path = PROJECT_ROOT / "data" / "evaluation" / "base_model_pred_fin_bert.csv"

fin_bert_predict(model=fin_bert_bl_model, data_loader=data_loader, path=ft_path)


# -

# ## SVR

# ### Set up dataset

# + pycharm={"name": "#%%\n"}
def setup_dataset(dataset_type: Type[L.LightningDataModule])-> tuple[pd.DataFrame, pd.DataFrame]:
    data_hand: L.LightningDataModule = dataset_type()
    data_hand.setup(stage="fit")

    svr_x: pd.DataFrame = data_hand.dataset.to_pandas()
    svr_y: pd.DataFrame = svr_x["label"]
    svr_x = svr_x.drop(columns=["id", "label", "spans"])

    # Define the custom function 'foo' that will transform each row in "embedder"
    # Assuming this function takes a list of 1D numpy arrays and outputs a numpy array of length K
    def foo(embedder_col):
        embeddings = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item) for item in embedder_col],
            batch_first=True
        ).cuda()
        aggregated_batch = torch.mean(embeddings, dim=1).cpu().numpy()
        return aggregated_batch


    # Step 1: Apply the custom function to the 'embedder' column and create a matrix of shape (N, K)
    embedder_col = svr_x["embedder"].apply(np.vstack)
    embedder_matrix = foo(embedder_col)

    # Step 2: Extract the other columns as a matrix of shape (N, Z)
    other_columns_matrix = svr_x.drop(columns=['embedder']).values

    # Step 3: Concatenate the two matrices (embedder_matrix and other_columns_matrix) along the second axis
    final_matrix = np.hstack((other_columns_matrix, embedder_matrix))

    return final_matrix, svr_y
# -

# ### Training

# + pycharm={"name": "#%%\n"}
svr_train_x, svr_train_y = setup_dataset(dm_hemlp.Semeval2017Train)

svr: BLBSSModel = BLBSSModel(model_type=BLBSSType.SVR, SVR_dataset=svr_train_x, SVR_labels=svr_train_y)

svr.fit()
# -

# ### Testing

# + pycharm={"name": "#%%\n"}
svr_test_x, svr_test_y = setup_dataset(dm_hemlp.Semeval2017Test)
svr_pred: np.ndarray[float] = svr.predict(svr_input={"X": svr_test_x})

svr_path = PROJECT_ROOT / "data" / "evaluation" / "base_model_pred_SVR.csv"

pred: pd.DataFrame = pd.DataFrame(svr_pred)
pred.to_csv(svr_path, index=False)
# -

# ## Evaluation

import src.utils.evaluation as eval_utils

svr_preds = pd.read_csv(svr_path)
ft_preds = pd.read_csv(ft_path)

# +
# TODO ( ͡° ͜ʖ ͡°) implement, take src/fine_tuned_finbert/evaluation/evaluate_finbert as example It is interesting
#  for us to evaluate every component of our ensemble so that we can see if we actually improved stuff at the end
import torchmetrics.classification as tc
from torch import Tensor

class SaharaEvaluator:
    def __init__(self, num_classes: int = 3):
        self._precision: tc.MulticlassPrecision = tc.MulticlassPrecision(num_classes=num_classes)
        self._recall: tc.MulticlassRecall = tc.MulticlassRecall(num_classes=num_classes)
        self._f1: tc.MulticlassF1Score = tc.MulticlassF1Score(num_classes=num_classes)

    # I put + 1 because pytorch recall, precision and f1 require non-negative tensors
    def precision(self, pred: Tensor, target: Tensor) -> Tensor:
        return self._precision(preds=pred + 1, target=target + 1)

    def recall(self, pred: Tensor, target: Tensor) -> Tensor:
        return self._recall(preds=pred + 1, target=target + 1)

    def f1(self, pred: Tensor, target: Tensor) -> Tensor:
        return self._f1(preds=pred + 1, target=target + 1)


# -

# ### SVR

svr_preds.head(20)

pd.DataFrame.from_records([{
    "recall": eval_utils.eval_fn_recall(svr_preds["0"].values, svr_test_y).aggregate_results["recall"],
    "cosine_similarity": eval_utils.eval_fn_cosine_similarity(svr_preds["0"].values, svr_test_y).aggregate_results["cosine_similarity"],
    "f1_score": eval_utils.eval_fn_f1(svr_preds["0"].values, svr_test_y).aggregate_results["f1"],
    "precision": eval_utils.eval_fn_precision(svr_preds["0"].values, svr_test_y).aggregate_results["precision"],
}])

# ### FinBERT

ft_preds.head(20)

# +
ft_y = data_handler.dataset.to_pandas()["label"].values

pd.DataFrame.from_records([{
    "recall": eval_utils.eval_fn_recall(ft_preds["0"].values, ft_y).aggregate_results["recall"],
    "cosine_similarity": eval_utils.eval_fn_cosine_similarity(ft_preds["0"].values, ft_y).aggregate_results["cosine_similarity"],
    "f1_score": eval_utils.eval_fn_f1(ft_preds["0"].values, ft_y).aggregate_results["f1"],
    "precision": eval_utils.eval_fn_precision(ft_preds["0"].values, ft_y).aggregate_results["precision"],
}])

# -


