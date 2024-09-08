from typing import NamedTuple


class VaderPosNegFeatures(NamedTuple):
    pos_neg_ratio: float
    pos_neg_difference: float


class ReadabilityMetrics(NamedTuple):
    flesch_kincaid_grade: float
    gunning_fog: float
    coleman_liau_index: float


class SentimentFeatures(NamedTuple):
    overall_valence_mean: float
    overall_arousal_mean: float
    overall_dominance_mean: float
    overall_valence_std: float
    overall_arousal_std: float
    overall_dominance_std: float
    valence_contrast: float
    arousal_contrast: float
    dominance_contrast: float
