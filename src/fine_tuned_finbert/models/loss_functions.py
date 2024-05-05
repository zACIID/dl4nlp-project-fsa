import torch
from torch import cosine_similarity
from torch.nn.functional import mse_loss, l1_loss


def custom_regression_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
) -> torch.Tensor:
    # loss = (-cosine_similarity(y_true, y_pred, dim=0) + 1) + l1_loss(y_true, y_pred)
    # loss = mse_loss(y_true, y_pred)
    loss = -cosine_similarity(y_true, y_pred, dim=-1)

    return loss


def sign_accuracy_mask(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return (((y_pred >= 0) & (y_true >= 0))
            | ((y_pred < 0) & (y_true < 0)))
