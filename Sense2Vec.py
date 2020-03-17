# This exercise is based on the following Sense2Vec blogpost:  https://explosion.ai/blog/sense2vec-with-spacy
#!/usr/bin/env python

import spacy
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import Input, Lambda, Dense, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

en = spacy.load('en_core_web_sm')

# Words that can be used as either a noun or (intransitive) verb
words = ['bank', 'battle', 'broadcast', 'camp', 'change', 'chant', 'dance', 'deal', 'design', 'dress', 'drill', 'drink', 'drive', 'email', 'trade', 'whistle']

# Window size for Word2Vec algorithm
window_size = 2

# Embedding size for Word2Vec algorithm
embedding_size = 7

def build_corpus(nouns_and_verbs = words):
    """
    Build the corpus used to train word embeddings
    Args:
      nouns_and_verbs - A list of words that can be used as either intransitive verbs or nouns
    Returns:
      A list of sentences
    """
    corpus = []
    for x in nouns_and_verbs:
        # Make sure that words appear in contexts for both verbs and nouns
        corpus.append("They like to {} quickly".format(x))
        corpus.append("They like the {} today".format(x))
    return corpus

def tag_nouns_and_verbs(sentence, tag_list=words):
    """
    Tag nouns and verbs in a list with their part-of-speech
    Args:
      sentence - the sentence to tag
      tag_list - list of words to tag with their part-of-speech
    Returns:
      The original sentence with words in the tag list in the format WORD>NOUN or WORD>VERB
    """
    doc = en(sentence)
    tokens = []
    for token in doc:
        if (token.text in tag_list) & (doc[-1] == 'quickly'):
            tokens.append(token.text.upper() + '>VERB')
        
        elif (token.text in tag_list) & (doc[-1] == 'today'):
            tokens.append(token.text.upper() + '>NOUN')
        ##TODO## : if a word is in the tag list and is a noun or verb, add it as WORD>NOUN or WORD>VERB
        else:
            tokens.append(token.text)
    return ' '.join(tokens)

def run_training(num_classes, X, y):
    """
    Perform the training run
    Args:
      num_classes - number of classes for the labels
      X - ground truth data
      y - ground truth labels
    """
    inputs = Input((window_size * 2,))

     ##TODO##: Complete embedding_layer code
    embedding_layer = Embedding(len(words) + 1 , embedding_size, input_length=len(inputs), name='embedding_layer')
  
    ##TODO##: Complete mean_layer code
    mean_layer = Lambda(lambda x: K.mean(x, axis=1))
    
    ##TODO##: Complete output layer code
    output_layer = Dense(2, activation='softmax')
    
    
 
    output = embedding_layer(inputs)
    output = mean_layer(output)
    output = output_layer(output)

    model = Model(inputs=[inputs], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.1, rho=0.99), metrics=['accuracy'])

    model.fit(X, y, batch_size=16, epochs=170, validation_split=0.1, verbose=2)
    
    return embedding_layer.get_weights()[0]

def make_cbow_data(sequences, window_size=window_size):
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

def main():

    # Build corpus with words that can be used as both nouns and verbs
    corpus = build_corpus()

    # Tag each of the words as either being a word or noun
    corpus = [tag_nouns_and_verbs(sentence) for sentence in corpus]
    
    # Generate numeric sequences with indices of the words in the corpus
    tokenizer = Tokenizer(lower=True, split=' ', filters='')
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)

    # Build (context, target) pairs for CBOW
    num_classes, cbow_X, cbow_y = make_cbow_data(sequences)

    # Run training
    embeddings = run_training(num_classes, cbow_X, cbow_y)

if __name__ == "__main__":
    main()
    K.clear_session()
