import torch


def custom_regression_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        k: float = 3.0,
        C: float = 1.0
) -> torch.Tensor:
    abs_error = torch.abs(y_true - y_pred)
    sign_mask = (~sign_accuracy_mask(y_true, y_pred)).int()  # Add penalty only if sign is wrong
    loss = (((k / (k - abs_error)) + C*sign_mask)**2 - 1).mean()
    # loss = (((k / (k - abs_error)) - 1) + C*sign_mask).mean() # TODO test this too -> main difference is that it should be flatter

    return loss


def sign_accuracy_mask(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return (((y_pred >= 0) & (y_true >= 0))
            | ((y_pred < 0) & (y_true < 0)))
