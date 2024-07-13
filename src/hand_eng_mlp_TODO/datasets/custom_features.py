
class CustomFeatures:
    def __init__(self):
        self.vader_polarity = 0.0

        self.pos_neg_ratio_vader = 0.0
        self.pos_neg_difference_vader = 0.0

        self.sentiment_entropy_vader = 0.0

        self.pos_neg_ratio_sentic = 0.0
        self.pos_neg_difference_sentic = 0.0

        self.swn_polarity = 0.0

        self.introspection = 0.0
        self.temper = 0.0
        self.attitude = 0.0
        self.sensitivity = 0.0

        self.flesch_kincaid_grade = 0.0
        self.gunning_fog = 0.0
        self.coleman_liau_index = 0.0

        self.overall_valence_mean = 0.0
        self.overall_arousal_mean = 0.0
        self.overall_dominance_mean = 0.0
        self.overall_valence_std = 0.0
        self.overall_arousal_std = 0.0
        self.overall_dominance_std = 0.0
        self.valence_contrast = 0.0
        self.arousal_contrast = 0.0
        self.dominance_contrast = 0.0

    def to_dict(self):
        return {
            'vader_polarity': self.vader_polarity,
            'pos_neg_ratio_vader': self.pos_neg_ratio_vader,
            'pos_neg_difference_vader': self.pos_neg_difference_vader,
            'sentiment_entropy_vader': self.sentiment_entropy_vader,
            'pos_neg_ratio_sentic': self.pos_neg_ratio_sentic,
            'pos_neg_difference_sentic': self.pos_neg_difference_sentic,
            'swn_polarity': self.swn_polarity,
            'introspection': self.introspection,
            'temper': self.temper,
            'attitude': self.attitude,
            'sensitivity': self.sensitivity,
            'flesch_kincaid_grade': self.flesch_kincaid_grade,
            'gunning_fog': self.gunning_fog,
            'coleman_liau_index': self.coleman_liau_index,
            'overall_valence_mean': self.overall_valence_mean,
            'overall_arousal_mean': self.overall_arousal_mean,
            'overall_dominance_mean': self.overall_dominance_mean,
            'overall_valence_std': self.overall_valence_std,
            'overall_arousal_std': self.overall_arousal_std,
            'overall_dominance_std': self.overall_dominance_std,
            'valence_contrast': self.valence_contrast,
            'arousal_contrast': self.arousal_contrast,
            'dominance_contrast': self.dominance_contrast
        }

    # Update methods
    def update_vader_polarity(self, vader_polarity):
        self.vader_polarity = vader_polarity

    def update_vader_pos_neg_features(self, pos_neg_ratio, pos_neg_difference):
        self.pos_neg_ratio_vader = pos_neg_ratio
        self.pos_neg_difference_vader = pos_neg_difference

    def update_vader_sentiment_entropy(self, sentiment_entropy):
        self.sentiment_entropy_vader = sentiment_entropy

    def update_emotions(self, emotions):
        self.introspection = emotions.get('INTROSPECTION', 0.0)
        self.temper = emotions.get('TEMPER', 0.0)
        self.attitude = emotions.get('ATTITUDE', 0.0)
        self.sensitivity = emotions.get('SENSITIVITY', 0.0)

    def update_sentic_pos_neg_features(self, pos_neg_ratio, pos_neg_difference):
        self.pos_neg_ratio_sentic = pos_neg_ratio
        self.pos_neg_difference_sentic = pos_neg_difference

    def update_swn_polarity(self, swn_polarity):
        self.swn_polarity = swn_polarity

    def update_readability_metrics(self, readability_metrics):
        self.flesch_kincaid_grade = readability_metrics[0]
        self.gunning_fog = readability_metrics[1]
        self.coleman_liau_index = readability_metrics[2]

    def update_overall_sentiment_features(self, overall_sentiment_features):
        self.overall_valence_mean = overall_sentiment_features[0]
        self.overall_arousal_mean = overall_sentiment_features[1]
        self.overall_dominance_mean = overall_sentiment_features[2]
        self.overall_valence_std = overall_sentiment_features[3]
        self.overall_arousal_std = overall_sentiment_features[4]
        self.overall_dominance_std = overall_sentiment_features[5]
        self.valence_contrast = overall_sentiment_features[6]
        self.arousal_contrast = overall_sentiment_features[7]
        self.dominance_contrast = overall_sentiment_features[8]
