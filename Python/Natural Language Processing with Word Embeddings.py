#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import HTML


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import HTML

corpus = ['The cat sat on the mat', 'The dog sat on the mat', 'The goat sat on the mat']

vectorizer = CountVectorizer(lowercase=True, analyzer='word', binary=False)
representation = vectorizer.fit_transform(corpus)
representation_df = pd.DataFrame(data = representation.toarray(), columns=sorted(vectorizer.vocabulary_.keys()))
representation_df


vectorizer = ##TODO## : Use CountVectorizer to create a binary value for each word
representation = vectorizer.fit_transform(corpus)
representation_df = pd.DataFrame(data = representation.toarray(), columns=sorted(vectorizer.vocabulary_.keys()))
representation_df


vectorizer = ##TODO## : Use CountVectorizer to remove English stopwords
representation = vectorizer.fit_transform(corpus)
representation_df = pd.DataFrame(data = representation.toarray(), columns=sorted(vectorizer.vocabulary_.keys()))
representation_df


ordering_corpus = ['The cat sat on the mat', 'the mat sat on the cat', 'Mat the cat the sat']
ordering_vectorizer = CountVectorizer(lowercase=True, analyzer='word', binary=True, stop_words='english')
representation = ordering_vectorizer.fit_transform(ordering_corpus)
representation_df = pd.DataFrame(data = representation.toarray(), columns=sorted(ordering_vectorizer.vocabulary_.keys()))
representation_df


vectorizer.vocabulary_


feature_corpus = ['The cat sat on the mat', 'The dog sat on the mat', 'The bird sat on the mat']
feature_vectorizer = CountVectorizer(lowercase=True, analyzer='word', binary=True, stop_words='english')
representation = feature_vectorizer.fit_transform(feature_corpus)
representation_df = pd.DataFrame(data = representation.toarray(), columns=sorted(feature_vectorizer.vocabulary_.keys()))
representation_df


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


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logistic = LogisticRegression()
y = allowed
X = representation_df[:len(y)]

logistic.fit(X,y) # Only train on the first 8 sentences
print("Training accuracy score is:  {} %".format(accuracy_score(logistic.predict(X), y)*100.0))


test_corpus = ['The keyboard sat on the mat', 'The bird sat on the mat']

rep = vectorizer.transform(test_corpus)

X_test =  rep
y_test = [0,1]
print("Expected Results for (keyboard, bird):  {}".format(y_test))
print("Actual   Results for (keyboard, bird):  {}".format(logistic.predict(X_test)))


import itertools

animals = ['cat','dog','goat','elephant','eagle','zebra','rhino', 'hippo']
actions = ['sat','stood','jumped','slept']
furniture = ['mat','rug','sofa','bed']

# Generate all combinations of animal, action and furniture
animal_corpus = ['The {} {} on the {}'.format(x[0], x[1], x[2]) for x in itertools.product(animals, actions, furniture)]
vocabulary_size = len(animals) + len(actions) + len(furniture) + 2

print("There are {} sentences in the corpus, with a vocabulary of {} words".format(len(animal_corpus), vocabulary_size))


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

# Generate (target, context) pairs with negative sampling

pairs = []
labels = []

for this_sequence in sequences:
    # Again note the vocabulary_size+1 expression
    c, l = skipgrams(this_sequence, vocabulary_size+1, window_size=WINDOW_SIZE, negative_samples=1, shuffle=True)
    for i in range(len(c)):
        pairs.append(c[i])
        labels.append(l[i])
pairs = np.array(pairs)
labels = np.array(labels)
    
print("There are {} (context,target) pairs in the dataset".format(len(pairs)))


from keras.layers import Embedding, Input, Dense, Reshape
from keras.layers.merge import Dot
from keras.models import Model
from keras.optimizers import RMSprop



target_word = Input((1,))
context_word = Input((1,))

# An embedding layer is just a lookup table - a matrix of size vocabulary_size x EMBEDDING_SIZE
# We select input_length rows from this matrix

 ##TODO## :  Add embedding layer nambed 'embedding_layer'.  Remember to add 1 to the vocabulary size!
embedding_layer = Embedding(vocabulary_size +1, EMBEDDING_SIZE, input_length=1, name='embedding_layer')

# Expect an output of similarity score between 0 and 1
output_layer = Dense(1, activation='sigmoid')

# Select the row indexed by target_word, reshape it for convenience
target_embedding = embedding_layer(target_word)
target_embedding = Reshape((EMBEDDING_SIZE,))(target_embedding)

# Select the row indexed by context_word, reshape it for convenience
context_embedding = embedding_layer(context_word)
context_embedding = Reshape((EMBEDDING_SIZE,))(context_embedding)


# Perform the dot product on the two embeddings, and run through the output sigmoid 
output = ##TODO## : Add dot product layer

output = output_layer(output)
    
# Setup a model for training
model = Model(inputs=[target_word, context_word], outputs=output)

optimizer = RMSprop(lr=0.0001, rho=0.99)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()


# Only print the results after this many epochs
INTERNAL_EPOCHS = 100

TOTAL_EPOCHS = 1500

def train(X,y):
    for index in range(int(TOTAL_EPOCHS / INTERNAL_EPOCHS)):
        h = model.fit(x=X, y=y, batch_size=256, epochs=INTERNAL_EPOCHS, verbose=0)
        print('Epoch {} - loss {}'.format((index + 1) * INTERNAL_EPOCHS, h.history['loss'][-1]))

train([pairs[:,0], pairs[:,1]], labels)        


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


category_colors = {'animals' : 'green', 
                   'actions' : 'blue',
                   'furniture' : 'yellow'}

colors = []
for i in range(vocabulary_size):
    colors.append('red')
    
for word in tokenizer.word_index:
    index = tokenizer.word_index[word] -1
    if word in animals:
        colors[index] = category_colors['animals']
    elif word in actions:
        colors[index] = category_colors['actions']
    elif word in furniture:
        colors[index] = category_colors['furniture']
        
def plot_embeddings_after_pca(vectors):  
        """
        Perform PCA and plot the resulting 2 components on X and Y axis
        Args:
          embedding_weights - the set of vectors to 
        """
        pca = PCA(n_components=2)
        vectors_pca = pca.fit_transform(vectors)
        plt.figure(figsize=(20,10))
        
        # We do not draw the first element, which is the 'Out-of-Vocabulary' token
        plt.scatter(vectors_pca[1:,0], vectors_pca[1:,1], c=colors, s=100, alpha=0.3);
        plt.title('Embeddings after PCA')
        legend_elements = [
                    plt.Line2D([0], [0], marker='o', color=category_colors['animals'], label='Animals'),
                    plt.Line2D([0], [0], marker='o', color=category_colors['actions'], label='Actions'),
                    plt.Line2D([0], [0], marker='o', color=category_colors['furniture'], label='Furniture'),
                    plt.Line2D([0], [0], marker='o', color='red', label='Other'),
                  ]

        # Create the figure
        plt.legend(handles=legend_elements);

        
plot_embeddings_after_pca(model.get_layer('embedding_layer').get_weights()[0])





from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import Lambda
import keras.backend as K

window_size = 2

def make_cbow_data(sequences, window_size):
    """
    Prepare CBOW data - given a sequence of words, return the set of subsequences of window_size words on the left and the right
    along with the 1-hot encoded context word
    Args:
      sequences - set of sequences that encode sentences
      window_size - the amount of words to look to the left and right of the context word
    Returns:
      num_classes - number of words in vocabulary
      X - numpy array of window_size words to the left and right of the context word
      y - 1-hot encoding of the context word
    """
    X = []
    y = []
    num_classes = len(np.unique(np.hstack(sequences)))+1
    for this_sequence in sequences:
        for output_index, this_word in enumerate(this_sequence):
            this_input = []
            y.append(np_utils.to_categorical(this_word, num_classes))
            input_indices = [output_index - i for i in range(window_size,0,-1)]
            input_indices += [output_index + i for i in range(1, window_size+1)]
            for i in input_indices:
                this_input.append(this_sequence[i] if i >= 0 and i < len(this_sequence) else 0)
            X.append(this_input)
    return num_classes, np.array(X),np.array(y)
                
   
        
num_classes, cbow_X, cbow_y = make_cbow_data(sequences, window_size)
print("cbow_X shape: {}".format(cbow_X.shape))
print("cbow_y shape: {}".format(cbow_y.shape))


inputs = Input((window_size * 2,))

embedding_layer = ##TODO## : Add the embedding layer code here
mean_layer = ##TODO## : Add the mean layer code here 
output_layer = ##TODO## : Add the output layer code here


output = embedding_layer(inputs)
output = mean_layer(output)
output = output_layer(output)

model = Model(inputs=[inputs], outputs=output)

optimizer = RMSprop(lr=0.1, rho=0.99)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


train(cbow_X, cbow_y)


plot_embeddings_after_pca(model.get_layer('embedding_layer').get_weights()[0])


from gensim.models import KeyedVectors

# Load embeddings from the pre-trained file
fastText_embeddings = KeyedVectors.load_word2vec_format('/dli/data/wiki/wiki.simple.vec')


def build_data_set(corpus):
    """
    Iterate over all sentences in the corpus and for each word, copy the embedding to the appropriate indices
    Args:
      corpus - list of individual sentences in the corpus
    Returns:
      X - Matrix of 1-hot encodings of the sentences
    """
    
    # Assume all sentences are of equal length - otherwise we'll need to truncate or pad
    words_in_sentence = len(corpus[0].split(' '))
    
    # Initialize 
    X = np.zeros((len(corpus), words_in_sentence * fastText_embeddings.vector_size))

    for sent_index in range(len(corpus)):
        words = corpus[sent_index].split(' ')
        for word_index in range(len(words)):
            start_index = fastText_embeddings.vector_size * word_index
            end_index = fastText_embeddings.vector_size * (word_index + 1)
            X[sent_index, start_index:end_index] = fastText_embeddings[words[word_index].lower()] 
    
    return X
            

# Train the model - animals are allowed on the mat, other objects are not
training_corpus = ['The cat sat on the mat', 'The dog sat on the mat', 'The goat sat on the mat', 'The elephant sat on the mat', 
          'The plane sat on the mat', 'The apple sat on the mat', 'The pen sat on the mat', 'The notebook sat on the mat']
allowed = np.array(
           [1,1,1,1,
            0,0,0,0])

X_train = build_data_set(training_corpus)
        
# Will the network be able to generalize?        
test_corpus = ['The keyboard sat on the mat', 'The bird sat on the mat']
X_test = build_data_set(test_corpus)


logistic_embeddings = LogisticRegression()
logistic_embeddings.fit(X_train, allowed)

y_test = [0,1]

print("Expected Results for (keyboard, bird):  {}".format(y_test))
print("Actual   Results for (keyboard, bird):  {}".format(logistic_embeddings.predict(X_test)))

