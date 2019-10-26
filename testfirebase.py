import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate(('C:/Users/Umarbek Nasimov/Desktop/yhack.json'))
firebase_admin.initialize_app(cred, {'databaseURL' : 'https://yhack-cb990.firebaseio.com/'})

root = db.reference()
# Add a new user under /users.
root.child('reviews').push({"Hi":2})
