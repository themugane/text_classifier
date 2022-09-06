# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
stemmer = LancasterStemmer()


# 3 classes of training data
training_data = []

training_data.append({"class": "positive", "sentence": "I love this car."})
training_data.append({"class": "positive", "sentence": "This view is amazing."})
training_data.append({"class": "positive", "sentence": "I feel great this morning."})
training_data.append({"class": "positive", "sentence": "I am so excited about the concert."})
training_data.append({"class": "positive", "sentence": "He is my best friend."})

training_data.append({"class": "negative", "sentence": "I do not like this car."})
training_data.append({"class": "negative", "sentence": "This view is horrible."})
training_data.append({"class": "negative", "sentence": "I feel tired this morning."})
training_data.append({"class": "negative", "sentence": "I am not looking forward to the concert."})
training_data.append({"class": "negative", "sentence": "He is my enemy."})


training_data.append({"class": "neutral", "sentence": "There is a book on the desk."})
training_data.append({"class": "neutral", "sentence": "Childhood is the time to play."})
training_data.append({"class": "neutral", "sentence": "All cows eat grass."})
training_data.append({"class": "neutral", "sentence": "what's for lunch?"})
training_data.append({"class": "neutral", "sentence": "I am Mr. Nimbus."})

print("%s sentences in training data" % len(training_data))