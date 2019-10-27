import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np 
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 


from collections import Counter


class Solution:
	def __init__(self):
		cred = credentials.Certificate(('C:/Users/Umarbek Nasimov/Desktop/yhack.json'))
		firebase_admin.initialize_app(cred, {'databaseURL' : 'https://yhack-cb990.firebaseio.com/'})

		self.ref = db.reference('reviews')
	def make_frequency_list(self):
		# Add a new user under /users.
		# for i in root.child('reviews'):
		#     print(i)
		review_dict = self.ref.get()
		self.word_freq = Counter()
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
		    text_string = ''.join(filter(whitelist.__contains__, text_string))

		    text_list = text_string.split(' ')
		    # for word in text_list:
		    # 	wordified = word.lower()
		    # 	bag_of_words.add(wordified)
		    for word in text_list:
		      wordified = word.lower()
		      if wordified not in stop_words:
		          self.word_freq[wordified] += 1
	def get_bag_of_words(self):
		self.bag_of_words = list(self.word_fred.keys())
	def averaged_perceptron(data, labels, params = {}, hook = None):
    def positive(x, th, th0):
        return np.sign(th.T@x + th0)
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    theta_sum = theta.copy()
    theta_0_sum = theta_0.copy()
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
            theta_sum = theta_sum + theta
            theta_0_sum = theta_0_sum + theta_0
    theta_avg = theta_sum / (T*n)
    theta_0_avg = theta_0_sum / (T*n)
    if hook: hook((theta_avg, theta_0_avg))
    return theta_avg, theta_0_avg


def get_x(text):





# snapshot = ref.order_by_child("rating").get()
# for key, val in snapshot.items():
#     print(key, val)

