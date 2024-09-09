import typing

import lightning
import pandas as pd
import torch
from utils.evaluation import (
    eval_fn_recall,
    eval_fn_precision,
    eval_fn_cosine_similarity,
    eval_fn_f1,
    apply_thresholds
)


def collate(embedder_col: pd.Series, new_features_cols: typing.List[pd.Series], model: lightning.LightningModule):
    embeddings = [torch.tensor(embedding.tolist()) for embedding in embedder_col]
    embeddings_batch = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

    new_features = [torch.tensor(feats.values) for feats in new_features_cols]
    new_features_batch = torch.stack(new_features).T

    return embeddings_batch, new_features_batch
