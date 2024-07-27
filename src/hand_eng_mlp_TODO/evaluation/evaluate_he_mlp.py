import torchmetrics.classification as tc
from torch import Tensor

# TODO: see their cosine similarity - https://alt.qcri.org/semeval2017/task5/index.php?id=evaluation
# TODO didn't understand shit about what I should do here

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
