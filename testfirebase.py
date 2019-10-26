import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords 


from collections import Counter

cred = credentials.Certificate(('./yhack.json'))
firebase_admin.initialize_app(cred, {'databaseURL' : 'https://yhack-cb990.firebaseio.com/'})

ref = db.reference('reviews')
# Add a new user under /users.
# for i in root.child('reviews'):
#     print(i)
review_dict = ref.get()
word_freq = Counter()
whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

stop_words = set(stopwords.words('english'))
stop_words.add("the")
stop_words.add("i")

print(stop_words)

for review in review_dict.values():
    text_string = review['text']
    text_string = ''.join(filter(whitelist.__contains__, text_string))

    text_list = text_string.split(' ')

    for word in text_list:
        if word not in stop_words:
            word_freq[word.lower()] += 1

# print(word_freq)


# snapshot = ref.order_by_child("rating").get()
# for key, val in snapshot.items():
#     print(key, val)

