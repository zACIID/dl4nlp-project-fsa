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


def collate(tokenizer_col: pd.Series, model: lightning.LightningModule):
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
