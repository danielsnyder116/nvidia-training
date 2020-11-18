#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from IPython.display import HTML


corpus = ['The cat sat on the mat', 'The dog sat on the mat', 'The goat sat on the mat']


#This countvectorizer takes care of a lot of preprocessing under the hood
# stop_words, tokenizing, lowercasing everything
vectorizer = CountVectorizer(lowercase=True, analyzer='word', binary=False)


#Fitting and transforming model
representation = vectorizer.fit_transform(corpus)

print(vectorizer.vocabulary_.keys())

df_rep = pd.DataFrame(data=representation.toarray(), columns=sorted(vectorizer.vocabulary_.keys()))

#Shows the frequency that each word shows up in the sentence
df_rep


#This also gets rid of stopwords
vectorizer = CountVectorizer(lowercase=True, analyzer='word', binary=True,
                             stop_words='english')


rep = vectorizer.fit_transform(corpus)

df_rep = pd.DataFrame(data=rep.toarray(), columns=vectorizer.vocabulary_.keys())


#This is a binary version, so no longer number of times the word shows up in sentence,
# but binary variable of whether or not sentence appears in each sentence
df_rep


vectorizer.vocabulary_


training_corpus = ['The cat sat on the mat', 'The dog sat on the mat', 'The goat sat on the mat', 'The elephant sat on the mat', 
          'The plane sat on the mat', 'The apple sat on the mat', 'The pen sat on the mat', 'The notebook sat on the mat']

allowed = [1,1,1,1,   # Objects that are allowed on the mat
           0,0,0,0]   # Objects that are not allowed on the mat

# Make sure that words we'll use in the test set are considered
for other_object in ['keyboard', 'bird']:
    training_corpus.append(other_object)   
    
vectorizer = CountVectorizer(lowercase=True, analyzer='word', binary=True, stop_words='english')
representation = vectorizer.fit_transform(training_corpus)
representation_df = pd.DataFrame(data = representation.toarray(), columns=sorted(vectorizer.vocabulary_.keys()))
representation_df


#Text Classification Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logr = LogisticRegression()

y = allowed
X = representation_df[:len(y)]

logr.fit(X,y)

print("Training Accuracy Score:{} %".format(accuracy_score(logr.predict(X), y)*100))


#Now that we've fit our model, we want to test it on new words.
#Since we've already fit our vectorizer, we now just use .transform 
#to convert the new strings to a one-hot encoded matrix

test_corpus = ['The keyboard sat on the mat', 'The bird sat on the mat']

rep = vectorizer.transform(test_corpus)

X_test =  rep
y_test = [0,1]
print("Expected Results for (keyboard, bird):  {}".format(y_test))
print("Actual   Results for (keyboard, bird):  {}".format(logr.predict(X_test)))


from itertools import product


animals = ['cat','dog','goat','elephant','eagle','zebra','rhino', 'hippo']
actions = ['sat','stood','jumped','slept']
furniture = ['mat','rug','sofa','bed']

# Generate all combinations of animal, action and furniture
animal_corpus = ['The {} {} on the {}'.format(x[0], x[1], x[2]) for x in itertools.product(animals, actions, furniture)]
vocabulary_size = len(animals) + len(actions) + len(furniture) + 2

print("There are {} sentences in the corpus, with a vocabulary of {} words".format(len(animal_corpus), vocabulary_size))

#So you can use product to get every single combination of three words
list(product(animals, actions, furniture))


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
import numpy as np

# Hyper-parameters

EMBEDDING_SIZE = 7  # Small corpus, so we're using a small dimension
WINDOW_SIZE = 4     # Empirically found to work well

# Convert text to numerical sequences

# Note that the Tokenizer starts numbering words with 1.  So we have vocabulary_size+1 words.  The 0-th word
# is considered to be the 'Out-of-vocabulary' token.
tokenizer = Tokenizer(num_words=vocabulary_size+1, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ',)
tokenizer.fit_on_texts(animal_corpus)
sequences = tokenizer.texts_to_sequences(animal_corpus)


sequences


np.hstack(sequences)




