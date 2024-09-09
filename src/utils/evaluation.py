import numpy as np
from mlflow.metrics import MetricValue
from sklearn.metrics import precision_score, recall_score, f1_score


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

