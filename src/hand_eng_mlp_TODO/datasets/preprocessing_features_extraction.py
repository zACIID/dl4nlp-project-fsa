import pandas as pd
import requests
import re
import textstat
import math
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from scipy.stats import entropy
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# TODO ( ͡° ͜ʖ ͡°) Ruie: we need to install these two, is there a file of python packages requirement?
# !pip install vaderSentiment
# !pip install textstat

nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Initialize sentiment analyzer globally
analyzer = SentimentIntensityAnalyzer()


def compute_sentence_polarity_vader(text):
    vader_score = analyzer.polarity_scores(text)['compound']  # compound is for overall score
    return vader_score


def emotion_recognition_sn(text):
    api_key = 'GqUQ3m0uJiWPD'
    url = f"http://sentic.net/api/en/{api_key}.py?text={text}"
    response = requests.get(url)
    if response.status_code == 200:
        # Extract emotion features from response
        match = re.search(r'\[(INTROSPECTION=.+?)\]', response.text)
        if match:
            features_text = match.group(1)
            # Extract individual emotion values
            features = {}
            for feature in features_text.split(','):
                name, value = feature.split('=')
                # Remove percentage symbol and convert to float
                value = float(value.rstrip('%')) / 100
                features[name] = value
            return features
        else:
            return None
    else:
        print("Error: Unable to retrieve emotion features from API")
        return None


# Function to calculate sentiment lexicon-based features with VADER:
# - ratio of positive to negative polarity words
# - difference between positive and negative words (normalized by total number of words)
def calculate_pos_neg_features_vader(text):
    scores = analyzer.polarity_scores(text)
    total_words = len(word_tokenize(text))
    # print(scores)

    if scores['neg'] != 0:
        pos_neg_ratio = scores['pos'] / scores['neg']
    else:
        pos_neg_ratio = scores['pos']

    if total_words != 0:
        pos_neg_difference = (scores['pos'] - scores['neg']) / total_words
    else:
        pos_neg_difference = 0

    return pos_neg_ratio, pos_neg_difference


# Function to calculate sentiment entropy using VADER
def calculate_sentiment_entropy_vader(text):
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

    return sentiment_entropy


# Function to fetch polarity of a word from SenticNet
def get_word_polarity(word):
    key = "u59p0l9yRM3Fk"
    url = f"http://sentic.net/api/en/{key}.py?text={word}"
    response = requests.get(url)
    if response.status_code == 200:
        polarity = response.text.strip()
        return polarity
    else:
        return None


# Function to calculate sentiment lexicon-based features with SenticNet:
# - ratio of positive to negative polarity words
# - difference between positive and negative words (normalized by total number of words)
def calculate_pos_neg_features_sn(text):
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

    if negative_words != 0:
        pos_neg_ratio_lexicon = positive_words / negative_words
    else:
        pos_neg_ratio_lexicon = positive_words

    if len(words) != 0:
        pos_neg_difference_lexicon = (positive_words - negative_words) / len(words)
    else:
        pos_neg_difference_lexicon = 0

    return pos_neg_ratio_lexicon, pos_neg_difference_lexicon


# Function to compute the overall polarity of a sentence using SentiWordNet
def compute_sentence_polarity_swn(text):
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
    if len(words) > 0:
        overall_polarity = (pos_score - neg_score) / len(words)
    else:
        overall_polarity = 0

    return overall_polarity


# Calculate readability metrics
def calculate_readability_metrics(text):
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    coleman_liau_index = textstat.coleman_liau_index(text)

    return flesch_kincaid_grade, gunning_fog, coleman_liau_index


# Lexicon with ratings for valence (pleasantness), arousal (intensity), and dominance (control)
# Ratings on three dimensions using a 9-point scale.
# If you feel completely neutral, neither happy nor sad [not excited nor at all calm; neither in control nor controlled], select the middle of the scale (rating 5).
# 1 (unhappy, calm, controlled) to 9 (happy, excited, in control)
# mean, std, level of contrast in terms of affect infused into the tweet
# V.Mean.Sum
# A.Mean.Sum
# D.Mean.Sum
def load_sentiment_dataset(dataset_path):
    # Load the "Norms of valence, arousal, and dominance for 13,915 English lemmas" dataset
    sentiment_data = pd.read_csv(dataset_path)

    # Compute the median of the three features
    valence_median = sentiment_data['V.Mean.Sum'].median()
    arousal_median = sentiment_data['A.Mean.Sum'].median()
    dominance_median = sentiment_data['D.Mean.Sum'].median()
    # print(f"Medians: V:{valence_median} A:{arousal_median} d:{dominance_median}")

    return sentiment_data, (valence_median, arousal_median, dominance_median)


def compute_overall_sentiment_features(text, sentiment_data, default_mean_value):
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

    return overall_valence_mean, overall_arousal_mean, overall_dominance_mean, overall_valence_std, overall_arousal_std, overall_dominance_std, valence_contrast, arousal_contrast, dominance_contrast


def create_example_dataset():
    data = {
        'text': [
            "This stock is performing exceptionally well, I'm optimistic about its future prospects.",
            "The company's financial results were disappointing, leading to a decline in investor confidence.",
            "My wallet got stolen, i'm extremely sad.",
            "Today i won the lottery, i'm super happy."
        ]
    }
    return pd.DataFrame(data)
