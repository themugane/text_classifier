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


words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)