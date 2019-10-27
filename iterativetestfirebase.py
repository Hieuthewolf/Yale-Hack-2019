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
        self.word_freq = Counter() # Mapping from words to their frequency in reviews.text

        self.word_to_avg_rating_mapping = dict() #Mapping from words to a (total rating, total review count) tuple

        self.text_to_word_mapping = dict() #Mapping from a text to a set of its filtered words (No stopwords)

        self.relevant_word_rating_mapping = dict() #Mapping from words to their average ratings for words that have over 100 occurences

        self.top_positive_words = set()
        self.top_negative_words = set()

        self.sorted_avg_list = []

    def make_frequency_dict(self):
        review_dict = self.ref.get()
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        stop_words = set(stopwords.words('english'))
        stop_words.add("the")
        stop_words.add("i")
        stop_words.add("us")
        stop_words.add("")

        stop_words = {word.lower() for word in stop_words}
        # bag_of_words = set()
        for review in review_dict.values():
            text_string = review['text']
            cleaned_text_string = ''.join(filter(whitelist.__contains__, text_string))

            text_list = cleaned_text_string.split(' ')

            for word in text_list:
                wordified = word.lower()
                if wordified not in stop_words:
                    self.word_freq[wordified] += 1

                self.text_to_word_mapping.setdefault(text_string, set()).add(wordified)

    def make_word_to_avg_rating_dict(self):
        review_dict = self.ref.get()
        for review in review_dict.values():
            rating = review['rating']
            text_string = review['text']

            for word in self.text_to_word_mapping[text_string]:
                if word not in self.word_to_avg_rating_mapping:
                    self.word_to_avg_rating_mapping[word] = (int(rating), 1)
                else:
                    total_rating, total_num_reviews = self.word_to_avg_rating_mapping[word]
                    self.word_to_avg_rating_mapping[word] = (total_rating + rating, total_num_reviews + 1)

    def make_relevant_word_rating_dict(self):
        for word, avg_rating_tuple in self.word_to_avg_rating_mapping.items():
            total_word_ratings = avg_rating_tuple[0]
            total_text_occurence = avg_rating_tuple[1]
            if total_text_occurence >= 75:
                self.relevant_word_rating_mapping[word] = total_word_ratings / total_text_occurence
        
        self.sorted_avg_list = sorted(list(self.relevant_word_rating_mapping.items()), key = lambda x: x[1])
        
    def get_frequency_dict(self):
        return self.word_freq

    def get_word_to_avg_rating_dict(self):
        return self.word_to_avg_rating_mapping

    def get_text_to_word_dict(self):
        return self.text_to_word_mapping

    def get_relevant_word_dict(self):
        return self.relevant_word_rating_mapping

    def get_top_rated_words(self):
        last_idx = len(self.sorted_avg_list) - 50
        return self.sorted_avg_list[last_idx:]

    def get_lowest_rated_words(self):
        return self.sorted_avg_list[:50]

if __name__ == "__main__":
    solution = Solution()
    solution.make_frequency_dict()
    solution.make_word_to_avg_rating_dict()
    solution.make_relevant_word_rating_dict()

    freq_dict = solution.get_frequency_dict()
    word_avg_dict = solution.get_word_to_avg_rating_dict()
    text_word_dict = solution.get_text_to_word_dict()
    relevant_word_dict = solution.get_relevant_word_dict()

    top_rated = solution.get_top_rated_words()
    bottom_rated = solution.get_lowest_rated_words()

    print(top_rated)
    print(bottom_rated)



