# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
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

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
# print([stemmer.stem(word.lower()) for word in w])
# print(training[i])


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2