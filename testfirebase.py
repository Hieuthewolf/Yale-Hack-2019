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
		self.data = []
		self.labels = []
		self.make_frequency_list()
		#print("this is freq list " + str(self.word_freq))
		self.bag_of_words()
		#print(self.new_data)
		print("running perc")
		self.run_perc()
		print(self.theta_avg, self.theta_0_avg)
	def make_frequency_list(self):
		# Add a new user under /users.
		# for i in root.child('reviews'):
		#     print(i)
		self.review_dict = self.ref.get()
		self.word_freq = Counter()
		whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

		stop_words = set(stopwords.words('english'))
		stop_words.add("the")
		stop_words.add("i")
		stop_words.add("us")
		stop_words.add("")

		stop_words = {word.lower() for word in stop_words}
		# bag_of_words = set()
		for review in self.review_dict.values():
			if int(review['rating'])<=2 or int(review['rating'])>=4:
				text_string = review['text']
				text_string = ''.join(filter(whitelist.__contains__, text_string))
				text_list = text_string.split(' ')
				for word in text_list:
					wordified = word.lower()
					if wordified not in stop_words:
						self.word_freq[wordified] += 1
		#print(self.word_freq)
		#get rid of words w freq less than 10
		self.new_freq = {}
		for word in self.word_freq:
			if self.word_freq[word]>=100: #can change this later
				self.new_freq[word]=self.word_freq[word]
		self.word_freq = self.new_freq 

		#print(self.word_freq)
		for review in self.review_dict.values():
			if int(review['rating'])<=2 or int(review['rating'])>=4:
				temp_words = set()
				text_string = review['text']
				text_string = ''.join(filter(whitelist.__contains__, text_string))
				text_list = text_string.split(' ')
				for word in text_list:
					wordified = word.lower()
					if wordified not in stop_words and wordified in self.word_freq:
						temp_words.add(wordified)
				if int(review['rating'])<=2:
					rating = -1
				else:
					rating = 1
				self.data.append(list(temp_words))
				self.labels.append(rating)
	def bag_of_words(self):
		self.bag_of_words = list(self.word_freq.keys())
		self.new_data = []
		for i in range(len(self.data)):
			curr_words = set(self.data[i])
			temp_result = []
			for word in self.bag_of_words:
				if word in curr_words:
					temp_result.append(1)
				else:
					temp_result.append(-1)
			self.new_data.append(temp_result)
	def run_perc(self):
		W = np.array(self.new_data).T
		labels = np.array([self.labels])
		#print(W[0:10,:],labels[:,0:10])
		#print(W.shape,labels.shape)
		self.averaged_perceptron(W,labels,100)
	def averaged_perceptron(self, data, labels, params = None):
	  def positive(x, th, th0):
	    return np.sign(th.T@x + th0)
	  if not params:
	  	T = 100
	  else:
	  	T = params
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
	  		theta_sum = theta_sum + theta
	  		theta_0_sum = theta_0_sum + theta_0
	  theta_avg = theta_sum / (T*n)
	  theta_0_avg = theta_0_sum / (T*n)
	  self.theta_avg, self.theta_0_avg = theta_avg,theta_0_avg

test = Solution()





# snapshot = ref.order_by_child("rating").get()
# for key, val in snapshot.items():
#     print(key, val)

