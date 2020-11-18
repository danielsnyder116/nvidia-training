#!/usr/bin/env python
# coding: utf-8
Rank the translations from Best to Worst

Rank: A E B D C
from keras.layers import Embedding, Bidirectional
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers.merge import Multiply
from keras.utils import to_categorical
from keras.layers import TimeDistributed
from keras.backend import int_shape

import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
import pandas as pd

# Setup multiple versions of OpenSeq2Seq

import sys
from importlib import import_module

sys.path.append('old_s2s')
old_run = import_module('run')

sys.path.remove('old_s2s')
sys.path.append('OpenSeq2Seq')
new_run = import_module('run2')

def run_open_seq2seq(version, args):
    args = args.split(' ')
    args.insert(0, '')
    sys.argv = args
    version.main()
    
    
get_ipython().run_line_magic('matplotlib', 'inline')


run_open_seq2seq(
    old_run, 
    '--config_file=old_s2s/example_configs/nmt_noatt.json --logdir=/dli/data/noatt --mode=infer --inference_out=baseline.txt')


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = create_dataset(m)

# Maximum sequence length
max_sequence_length = 30


sources, targets = zip(*dataset)
sources = np.array([string_to_int(i, max_sequence_length, human_vocab) for i in sources])
targets = [string_to_int(t, max_sequence_length, machine_vocab) for t in targets]
targets = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), targets)))


index = np.random.randint(m)

print("sources shape          : ", sources.shape)
print("targets shape          : ", targets.shape)
print("Length of human_vocab  : ", len(human_vocab))
print("Length of machine_vocab: ", len(machine_vocab))

print("\n")
print("Human-readable date                 : ", dataset[index][0])
print("Machine-readable date               : ", dataset[index][1])
print("Pre-processed human-readable date   : \n", sources[index])
print("Pre-processed machine-readable date : \n", targets[index]) # 1-hot encoded

Before continuing, make sure you understand how the data is encoded.  What happens when the result sequence is shorter than 20 characters?

Answer:  The result sequence is then padded to be at least 20 characters long.
def model_simple_nmt(human_vocab_size, machine_vocab_size, max_sequence_length = 20):
    """
    Simple Neural Machine Translation model
    
    Arguments:
    human_vocab_size -- size of the human vocabulary for dates, it will give us the size of the embedding layer
    machine_vocab_size -- size of the machine vocabulary for dates, it will give us the size of the output vector
    
    Returns:
    model -- model instance in Keras
    """
    
    ### START CODE HERE ###
    # Define the input of your model with a shape (max_sequence_length,)
    
    inputs = Input((max_sequence_length,), name='input_layer')
    
    # Define the embedding layer. Embedding dimension should be 64 and input_length set to max_sequence_length.
    input_embed = Embedding(human_vocab_size, 64, input_length=max_sequence_length, name='Embedding_Layer')(inputs)
    
    # Encode the embeddings using a bidirectional LSTM
    enc_out = Bidirectional(LSTM(32, return_sequences=True), name='Encoder')(input_embed)
    
    # Decode the encoder output using an LSTM layer
    dec_out = LSTM(32, name='Decoder', return_sequences=True)(enc_out)
    
    ### END CODE HERE ###
    
    # Apply Dense layer to every time step
    output = TimeDistributed(Dense(machine_vocab_size, activation='softmax'))(dec_out)
    
    # Create model instance 
    model = Model(input=[inputs], output=output)

    return model


# Create model
model = model_simple_nmt(len(human_vocab), len(machine_vocab), max_sequence_length)

# Compile model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# See the model summary
model.summary()


model.fit([sources[:8000]], targets[:8000], epochs=40, batch_size=128, validation_split=0.1, verbose=2)


_,accuracy = model.evaluate([sources[8000:]], targets[8000:], batch_size=128,verbose=1)


print(accuracy)


EXAMPLES1 = ['03.3.2000', '03 03 2000', '03 mar 2000', '03 march 2000', 'march 03 2000', 'march 03, 2000' ]
EXAMPLES2 = ['04.4.2001', '04 04 2001', '03 apr 2000', '04 april 2001', 'april 04 2001', 'april 04, 2001' ]

def prediction_to_text(model_results):
    results = np.argmax(model_results[0], axis=-1)
    output = int_to_string(results, inv_machine_vocab)
    return ''.join(output).replace('<pad>','')

def run_examples(examples):
    for example in examples:
        print("source:", example)
        source = string_to_int(example, max_sequence_length, human_vocab)
        prediction = model.predict(np.array([source]))
        print("output:", prediction_to_text(prediction))
        print('******')
    
run_examples(EXAMPLES1)
print('---------')
run_examples(EXAMPLES2)


# Import all data into a single data frame for convenience
df = pd.DataFrame(data=dataset[:8000], columns=['Source','Target'])
df['Source Length'] = df['Source'].apply(lambda x: len(x))

# Have the model predict all training data
model_predictions = model.predict(sources[:8000])
df['Actual'] = [prediction_to_text([x]) for x in model_predictions]

# Check if prediction matches ground truth
df['Is Correct'] = df['Actual'] == df['Target']

df.head()


if not df['Is Correct'].any():
    print("Please re-train the model!")
else:
    ax = pd.crosstab(df['Source Length'], df['Is Correct'], normalize='index')[True].plot(kind='line', figsize=(20,7))
    ax.set_ylabel("% Correct")
    ax.set_xlabel("Input Length (characters)");


day_location = 8 #  Recall that the machine-readable format is YYYY-MM-DD.  We look for the index where DD begins.


location_data = []
for x in df.iterrows():
    source = x[1].Source
    day_value = x[1].Target[day_location:day_location+2]
    if day_value[0] == '0':
        day_value = day_value[1]
    day_index = source.find(day_value)
        
    location_data.append((day_index))
    
location_data = np.array(location_data)

df['Day Location'] = location_data


df.head()


if not df['Is Correct'].any():
    print("Please re-train the model!")
else:
    ax = pd.crosstab(df['Day Location'], df['Is Correct'], normalize='index')[True].plot(kind='line', figsize=(20,7))
    ax.set_ylabel("% Correct")
    ax.set_xlabel("Location of Day Component");


# Generate the Spearman correlation coefficient between the input source length and location of the day component.
print("Spearman correlation coefficient: ", np.corrcoef(df['Source Length'], df['Day Location'])[0][1])


import os
if not os.path.exists('./dates'):
    os.makedirs('./dates')


def save_data(filename, data):
    with open('dates/' + filename + '.txt', 'w') as f:
        for x in data:
            f.write(x)
            f.write("\n")
            
save_data('source_train', map(lambda x: '|'.join(x[0]), dataset[:8000]))
save_data('target_train', map(lambda x: '|'.join(x[1]), dataset[:8000]))
save_data('source_test', map(lambda x: '|'.join(x[0]), dataset[8000:]))
save_data('target_test', map(lambda x: '|'.join(x[1]), dataset[8000:]))
save_data('source_vocab', [key for key,value in sorted(human_vocab.items(), key=lambda x: x[1])])
save_data('target_vocab', [key for key,value in sorted(machine_vocab.items(), key=lambda x: x[1])])


get_ipython().system('rm -rf ./dates_log')


get_ipython().run_cell_magic('time', '', "\nrun_open_seq2seq(\n    new_run, \n    '--config_file=dates_config.py --mode=train')")


get_ipython().run_cell_magic('time', '', "\nrun_open_seq2seq(\n    new_run, \n    '--config_file=dates_config.py --mode=infer')")


df = pd.concat([
    pd.DataFrame(dataset[8000:], columns=['Source', 'Target']), 
    pd.read_csv('infer-out.txt', names=['Actual'], header=None) 
], axis=1)
df['Actual'] = df['Actual'].str.replace(' ','')
df['Is Correct'] = df['Actual'] == df['Target']
df['Source Length'] = df['Source'].apply(lambda x: len(x))



print("Accuracy: {}".format(df['Is Correct'].sum() / len(df)))
df.head()


if not df['Is Correct'].any():
    print("Please re-train the model!")
else:
    ax = pd.crosstab(df['Source Length'], df['Is Correct'], normalize='index')[True].plot(kind='line', figsize=(20,7))
    ax.set_ylabel("% Correct")
    ax.set_xlabel("Input Length (characters)");


get_ipython().system('rm -rf ./dates_eval_log')


get_ipython().run_cell_magic('time', '', "\nrun_open_seq2seq(\n    new_run, \n    '--config_file=dates_eval_config.py --mode=train_eval')")


get_ipython().system('head /dli/data/wmt/newstest2015.tok.bpe.32000.de')


run_open_seq2seq(
    old_run, 
    '--config_file=old_s2s/example_configs/nmt.json --logdir=/dli/data/nmt --mode=infer --inference_out=pred.txt')


# Test your implementation
# Kernel->restart will ensure the latest version of my_bleu.py is imported if you made any changes to the code to re-test

import my_bleu as mb

output1 = '2007-07-11<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
expected = '2007-07-11'

print(mb.str_score(output1, expected))    #expecting 1.0
print(mb.char_score(output1, expected)) # expecting 1.0

output2 = '2007-07-12<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>' #it's 12 instead of 011

print(mb.str_score(output2, expected))    #expecting 0.7598356856515925
print(mb.char_score(output2, expected)) #expecting 0.8801117367933934


m = model_simple_nmt(len(human_vocab), len(machine_vocab))

m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(m.summary())


inputs, targets = zip(*dataset)
inputs = np.array([string_to_int(i, 20, human_vocab) for i in inputs])
targets = [string_to_int(t, 20, machine_vocab) for t in targets]
targets = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), targets)))


m.fit([inputs], targets, epochs=1, batch_size=64, validation_split=0.1)


def keras_rnn_predict(samples, empty=human_vocab["<pad>"], rnn_model=m, maxlen=30):
    """for every sample, calculate probability for every possible label
    you need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty)
    return rnn_model.predict(data, verbose=0)


def beamsearch(predict=keras_rnn_predict, k=1, maxsample=10, 
               use_unk=False, 
               oov=human_vocab["<unk>"], 
               empty=human_vocab["<pad>"], 
               eos=human_vocab["<unk>"]):
    """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
    all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
    """
    
    dead_k = 0 # samples that reached eos
    dead_samples = []
    dead_scores = []
    live_k = 1 # samples that did not yet reached eos
    live_samples = [[empty]]
    live_scores = [0]

    while live_k and dead_k < k:
        # for every possible live sample calc prob for every possible label 
        probs = predict(live_samples, empty=empty)

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        if not use_unk and oov is not None:
            cand_scores[:,oov] = 1e20
        cand_flat = cand_scores.flatten()

        # find the best (lowest) scores we have from all possible samples and new words
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        live_scores = cand_flat[ranks_flat]

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        live_samples = [live_samples[r//voc_size]+[r%voc_size] for r in ranks_flat]

        # live samples that should be dead are...
        zombie = [s[-1] == eos or len(s) >= maxsample for s in live_samples]
        
        # add zombies to the dead
        dead_samples += [s for s,z in zip(live_samples,zombie) if z]  # remove first label == empty
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living 
        live_samples = [s for s,z in zip(live_samples,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]
        live_k = len(live_samples)

    return live_samples + dead_samples, live_scores + dead_scores

