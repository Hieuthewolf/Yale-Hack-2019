import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

# nltk.download('averaged_perceptron_tagger')

import re

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np 
import nltk

from nltk.corpus import stopwords 

from collections import Counter

class Solution:
    def __init__(self):
        cred = credentials.Certificate(('./yhack.json'))
        firebase_admin.initialize_app(cred, {'databaseURL' : 'https://yhack-cb990.firebaseio.com/'})

        self.ref = db.reference('reviews')

        self.all_possible_words = [] # Contains all possible valid words that are nouns and adjectives in review
        self.reviews = [] # Contains a list of (review, pos/neg rating) tuples

        # J = adjective, N = noun
        self.allowed_parts_of_speech = ["J", "N"]

        self.frequency_threshold = 75 # frequency threshold to cap at a certain amount of words
        self.most_frequent_words = [] # Most frequently used words in review where frequent classified by >= 75 words
        self.word_feature_sets = [] # Contains all the word feature sets of every review where each feature set = essentially bag of words

        # Sk models used for ensemble classifiers to combine multiple learning algorithms
        self.MNB_clf = None # 
        self.BNB_clf = None
        self.LogReg_clf = None
        self.SGD_clf = None
        self.SVC_clf = None

        # Keeps track of the counts of each word based on sentiment
        self.wordSentimentCounter = Counter()

        # Stopwords to not include in all_possible_words
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.add("the")
        self.stop_words.add("i")
        self.stop_words.add("us")
        self.stop_words.add("")
        self.stop_words = {word.lower() for word in self.stop_words}

    def make_reviews(self):
        review_dict = self.ref.get()

        # bag_of_words = set()
        for review in review_dict.values():
            text_string = review['text']
            rating = int(review['rating'])

            if rating >= 4:
                self.reviews.append((text_string, "pos"))
            else:
                self.reviews.append((text_string, "neg"))

            clean_text = re.sub(r'[^(a-zA-Z)\s]', '', text_string)
            tokenized = word_tokenize(clean_text)
            # Lowercase all words in tokenize
            tokenized = [word.lower() for word in tokenized]

            clean_words = [word for word in tokenized if word not in self.stop_words]

            # parts of speech associated for each word
            parts_of_speech = nltk.pos_tag(clean_words)

            for word, part_of_speech in parts_of_speech:
                if part_of_speech[0] in self.allowed_parts_of_speech:
                    self.all_possible_words.append(word.lower())

    def create_freq_distribution(self):
        # Mutates self.all_possible_words from list to now a frequency dictionary
        self.all_possible_words = nltk.FreqDist(self.all_possible_words)

        # Gets the top 200 most frequent words
        most_frequent_words = []
        for word, freq in self.all_possible_words.items():
            if freq >= self.frequency_threshold:
                most_frequent_words.append(word)

        self.most_frequent_words = most_frequent_words

    """
    Creates a dictionary of features for each review in self.reviews
    Keys = words in most_frequent_words
    Value = each key = true or false whether the feature appears in review or not
    """
    def identify_word_features(self, review_text):
        words = word_tokenize(review_text)
        words_in_review = {}
        for word in self.most_frequent_words:
            words_in_review[word] = (word in words)
        return words_in_review

    # Creates word feature sets for every single review as described in method above
    def create_words_dict_for_all_reviews(self):    
        self.word_feature_sets = [(self.identify_word_features(review), rating_indicator) for (review, rating_indicator) in self.reviews]

    def train_models_with_sk_models(self):
        self.MNB_clf = SklearnClassifier(MultinomialNB())
        self.MNB_clf.train(self.word_feature_sets)

        self.BNB_clf = SklearnClassifier(BernoulliNB())
        self.BNB_clf.train(self.word_feature_sets)

        self.LogReg_clf = SklearnClassifier(LogisticRegression())
        self.LogReg_clf.train(self.word_feature_sets)

        self.SGD_clf = SklearnClassifier(SGDClassifier())
        self.SGD_clf.train(self.word_feature_sets)

        self.SVC_clf = SklearnClassifier(SVC())
        self.SVC_clf.train(self.word_feature_sets)

    """
    Using ensemble_clf of clf's and a CLEANED review_text --> generate a word_feature_set
    return --> the classification (positive, negative), confidence of that classification
    """
    def sentiment(self, ensemble_clf, review_text):
        word_feature_set = self.identify_word_features(review_text)
        return ensemble_clf.classify(word_feature_set), ensemble_clf.confidence(word_feature_set)

    def get_most_frequent_word_features(self):
        classifier = nltk.NaiveBayesClassifier.train(self.word_feature_sets)

        print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, self.word_feature_sets))*100)

        classifier.show_most_informative_features(60)

    def get_word_feature_sets(self):
        return self.word_feature_sets

    def build_word_sentiment_counter(self):
        for review in db.reference('reviews').get().values():
            # Retrieve text from review and do preprocessing regex
            text_string = review['text']
            clean_text = re.sub(r'[^(a-zA-Z)\s]', '', text_string)
            
            # Associated label (pos, neg) and confidence level of sentiment of the text using our ensemble
            label, confidence = solution.sentiment(ensemble_clf, clean_text)

            tokenized = word_tokenize(clean_text)
            tokenized = [word.lower() for word in tokenized]
            clean_words = [word for word in tokenized if word not in self.stop_words]

            for word in clean_words:
                if word in self.most_frequent_words:
                    if confidence > 0.5 and label == "pos":
                        self.wordSentimentCounter[word] += 1
                    elif confidence > 0.5 and label == "neg":
                        self.wordSentimentCounter[word] -= 1

    def get_word_setiment_counter(self):
        return self.wordSentimentCounter

from nltk.classify import ClassifierI

class EnsembleClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)
        return mode(votes)

    # a simple measurement the degree of confidence in the classification 
    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

if __name__ == "__main__":
    solution = Solution()
    solution.make_reviews() # Makes self.review and self.all_possible_words are filled in this method
    solution.create_freq_distribution() # Creates freq distribution using self.all_possible_words and mutating self.most_frequent_words
    solution.create_words_dict_for_all_reviews() # Creates all the word feature sets for all reviews in our dataset

    # Simply gets the most frequent words and its corresponding labels from specified word count 
    # solution.get_most_frequent_word_features()

    solution.train_models_with_sk_models() # Trains our models with the 5 sk training models

    # Initializing the ensemble classifier 
    ensemble_clf = EnsembleClassifier(solution.SVC_clf, solution.MNB_clf, solution.BNB_clf, solution.LogReg_clf, solution.SGD_clf)

    solution.build_word_sentiment_counter()
    print(solution.get_word_setiment_counter())






    



