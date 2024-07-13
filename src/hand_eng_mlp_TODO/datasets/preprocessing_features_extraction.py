import pandas as pd
import requests
import re
import textstat
import math
import nltk
import os
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from scipy.stats import entropy
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, Tuple, Union, List
from dotenv import load_dotenv
from custom_features import CustomFeatures
from utils.custom_features_utils import download_and_extract_zip

# The following two packages have been added to pyproject.toml
# poetry add vaderSentiment
# poetry add textstat

# TODO update docstrings

load_dotenv()
SENTICNET_API_EMOTION_KEY = os.getenv('SENTICNET_API_EMOTION_KEY')
SENTICNET_API_POLARITY_KEY = os.getenv('SENTICNET_API_POLARITY_KEY')

nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Initialize sentiment analyzer globally
analyzer = SentimentIntensityAnalyzer()


def compute_vader_polarity(text: str, features: CustomFeatures) -> CustomFeatures:
    """
    Computes the overall sentiment polarity of a sentence using VADER

    :param text:
    :return:
    """
    vader_score = analyzer.polarity_scores(text)['compound']
    features.update_vader_polarity(vader_score)
    return features


def calculate_vader_pos_neg_features(text: str, features: CustomFeatures) -> CustomFeatures:
    """
    Calculates sentiment lexicon-based features using VADER:
    - ratio of positive to negative polarity words
    - difference between positive and negative words (normalized by total number of words)
    :param text:
    :return: tuple containing the positive/negative ratio and the positive/negative difference.
    """
    scores = analyzer.polarity_scores(text)
    total_words = len(word_tokenize(text))
    # print(scores)

    pos_neg_ratio = scores['pos'] / scores['neg'] if scores['neg'] != 0 else scores['pos']
    pos_neg_difference = (scores['pos'] - scores['neg']) / total_words if total_words != 0 else 0
    features.update_vader_pos_neg_features(pos_neg_ratio, pos_neg_difference)
    return features


def calculate_vader_sentiment_entropy(text: str, features: CustomFeatures) -> CustomFeatures:
    """
    Calculates the sentiment entropy of a text using VADER.

    :param text:
    :return:
    """
    words = word_tokenize(text)
    # print(words)

    # Sentiment scores for each word
    sentiment_scores = [analyzer.polarity_scores(word)['compound'] for word in words]
    # Probability distribution of sentiment scores
    sentiment_counts = Counter(sentiment_scores)
    total_words = len(sentiment_scores)
    probabilities = [count / total_words for count in sentiment_counts.values()]
    # Entropy
    sentiment_entropy = entropy(probabilities)
    features.update_vader_sentiment_entropy(sentiment_entropy)
    return features


def sentic_emotion_recognition(text: str, features: CustomFeatures) -> CustomFeatures:
    """
    Extract emotions in text using SenticNet

    :param text:
    :return: A dictionary containing emotion features (INTROSPECTION, TEMPER, ATTITUDE, SENSITIVITY)
             with their respective float values, or None if the API call fails.
    """
    url = f"http://sentic.net/api/en/{SENTICNET_API_EMOTION_KEY}.py?text={text}"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.search(r'\[(INTROSPECTION=.+?)\]', response.text)
        if match:
            features_text = match.group(1)
            emotions = {}
            for feature in features_text.split(','):
                name, value = feature.split('=')
                value = float(value.rstrip('%')) / 100
                emotions[name] = value
            features.update_emotions(emotions)
    else:
        print("Error: Unable to retrieve emotion features from API")
    return features


def get_word_polarity(word: str) -> Union[str, None]:
    """
    Fetches the polarity of a word from SenticNet.

    :param word:
    :return: The polarity of the word ('POSITIVE', 'NEGATIVE', or None if not found).
    """
    url = f"http://sentic.net/api/en/{SENTICNET_API_POLARITY_KEY}.py?text={word}"
    response = requests.get(url)
    if response.status_code == 200:
        polarity = response.text.strip()
        return polarity
    else:
        return None


def calculate_sentic_pos_neg_features(text: str, features: CustomFeatures) -> CustomFeatures:
    """
    Calculates sentiment lexicon-based features with SenticNet.
    - ratio of positive to negative polarity words
    - difference between positive and negative words (normalized by total number of words)

    :param text: The input text string.
    :return: A tuple containing the positive/negative ratio and the positive/negative difference.
    """
    words = word_tokenize(text)
    # print(words)
    positive_words = 0
    negative_words = 0
    for word in words:
        polarity = get_word_polarity(word)
        if polarity == 'POSITIVE':
            positive_words += 1
        elif polarity == 'NEGATIVE':
            negative_words += 1

    pos_neg_ratio = positive_words / negative_words if negative_words != 0 else positive_words
    pos_neg_difference = (positive_words - negative_words) / len(words) if len(words) != 0 else 0

    features.update_sentic_pos_neg_features(pos_neg_ratio, pos_neg_difference)
    return features


def compute_swn_polarity(text: str, features: CustomFeatures) -> CustomFeatures:
    """
    Computes the overall polarity of a sentence using SentiWordNet.

    :param text:
    :return:
    """
    words = word_tokenize(text)

    # Variables to store positive and negative scores for the sentence
    pos_score = 0
    neg_score = 0

    for word in words:
        # Get the sentiment scores for the word
        word_synsets = list(swn.senti_synsets(word))
        if word_synsets:
            # Calculate the average positive and negative scores for the word
            pos_word_score = sum(synset.pos_score() for synset in word_synsets) / len(word_synsets)
            neg_word_score = sum(synset.neg_score() for synset in word_synsets) / len(word_synsets)
            # Update scores for the sentence
            pos_score += pos_word_score
            neg_score += neg_word_score

    # Overall polarity score for the sentence
    overall_polarity = (pos_score - neg_score) / len(words) if len(words) > 0 else 0

    features.update_swn_polarity(overall_polarity)
    return features


def calculate_readability_metrics(text: str, features: CustomFeatures) -> CustomFeatures:
    """
    Calculates readability metrics for a given text.

    :param text:
    :return: A tuple containing Flesch-Kincaid grade, Gunning fog index, and Coleman-Liau index.
    """
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    coleman_liau_index = textstat.coleman_liau_index(text)

    features.update_readability_metrics((flesch_kincaid_grade, gunning_fog, coleman_liau_index))
    return features


def load_sentiment_dataset(extract_to: str) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    """
    Loads the sentiment dataset and computes the median of valence, arousal, and dominance features.
    In particular, the sentiment dataset is a Lexicon with ratings for valence (pleasantness), arousal (intensity), and dominance (control).
    The Ratings are on three dimensions using a 9-point scale.
    1 (unhappy, calm, controlled) to 9 (happy, excited, in control), 5 if completely neutral

    :param extract_to: directory where the sentiment dataset CSV file will be extracted to.
    :return: DataFrame with sentiment data and tuple with medians of valence, arousal, and dominance.
    """
    dataset_url = "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip"
    download_and_extract_zip(dataset_url, extract_to)
    file_name = 'BRM-emot-submit.csv'
    dataset_path = os.path.join(extract_to, file_name)

    # check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The file {file_name} was not found in the extracted contents.")

    # Load the "Norms of valence, arousal, and dominance for 13,915 English lemmas" dataset
    sentiment_data = pd.read_csv(dataset_path)

    # Compute the median of the three features
    valence_median = sentiment_data['V.Mean.Sum'].median()
    arousal_median = sentiment_data['A.Mean.Sum'].median()
    dominance_median = sentiment_data['D.Mean.Sum'].median()
    # print(f"Medians: V:{valence_median} A:{arousal_median} d:{dominance_median}")

    return sentiment_data, (valence_median, arousal_median, dominance_median)


def compute_overall_sentiment_features(
        text: str,
        sentiment_data: pd.DataFrame,
        default_mean_value: Tuple[float, float, float],
        features: CustomFeatures
) -> CustomFeatures:
    """
    Computes overall sentiment features for a text based on valence, arousal, and dominance.

    :param text:
    :param sentiment_data: sentiment data DataFrame
    :param default_mean_value: default mean values for valence, arousal, and dominance
    :return: tuple with overall mean and std for valence, arousal, and dominance, and their respective contrasts.
    """
    words = word_tokenize(text)

    # Lists to store valence, arousal, and dominance values
    valence_values = []
    arousal_values = []
    dominance_values = []

    for word in words:
        word_data = sentiment_data[sentiment_data['Word'] == word]
        if not word_data.empty:
            # Get the valence, arousal, and dominance values for the word
            valence = word_data['V.Mean.Sum'].values[0]
            arousal = word_data['A.Mean.Sum'].values[0]
            dominance = word_data['D.Mean.Sum'].values[0]
            valence_values.append(valence)
            arousal_values.append(arousal)
            dominance_values.append(dominance)
            # print(f"word: {word}, scores: V:{valence} A:{arousal} D:{dominance}")
        else:
            # Get the default values for valence, arousal, and dominance for the word not in the lexicon
            valence, arousal, dominance = default_mean_value
            valence_values.append(valence)
            arousal_values.append(arousal)
            dominance_values.append(dominance)

    # Compute overall mean and standard deviation for valence, arousal, and dominance
    overall_valence_mean = sum(valence_values) / len(valence_values) if valence_values else 0
    overall_arousal_mean = sum(arousal_values) / len(arousal_values) if arousal_values else 0
    overall_dominance_mean = sum(dominance_values) / len(dominance_values) if dominance_values else 0

    # Standard deviation in the text
    overall_valence_std = pd.Series(valence_values).std() if valence_values else 0
    overall_arousal_std = pd.Series(arousal_values).std() if arousal_values else 0
    overall_dominance_std = pd.Series(dominance_values).std() if dominance_values else 0

    # Contrast in the text
    valence_contrast = max(valence_values) - min(valence_values)
    arousal_contrast = max(arousal_values) - min(arousal_values)
    dominance_contrast = max(dominance_values) - min(dominance_values)

    features.update_overall_sentiment_features((overall_valence_mean, overall_arousal_mean, overall_dominance_mean,
                                                overall_valence_std, overall_arousal_std, overall_dominance_std,
                                                valence_contrast, arousal_contrast, dominance_contrast))
    return features


def extract_all_features(
        text: str,
        sentiment_data: pd.DataFrame,
        default_mean_value: Tuple[float, float, float]
) -> Dict:
    """
    Extracts all features for a given text by calling all the feature extraction functions.

    :param text: input text string.
    :param sentiment_data: Sentiment data DataFrame.
    :param default_mean_value: Default mean values for valence, arousal, and dominance.
    :return: A dictionary containing all the extracted features.
    """
    features = CustomFeatures()

    # Call each feature extraction function
    features = compute_vader_polarity(text, features)
    features = calculate_vader_pos_neg_features(text, features)
    features = calculate_vader_sentiment_entropy(text, features)
    features = sentic_emotion_recognition(text, features)
    features = calculate_sentic_pos_neg_features(text, features)
    features = compute_swn_polarity(text, features)
    features = calculate_readability_metrics(text, features)
    features = compute_overall_sentiment_features(text, sentiment_data, default_mean_value, features)

    # Return the features as a dictionary
    return features.to_dict()
