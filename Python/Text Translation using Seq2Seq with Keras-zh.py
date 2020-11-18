#!/usr/bin/env python
# coding: utf-8
将翻译结果从最佳到最差排名

排名：填写
get_ipython().system('python3 OpenSeq2Seq/run.py --config_file=OpenSeq2Seq/example_configs/nmt.json --logdir=/dli/data/noatt --mode=infer --inference_out=baseline.txt')


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
get_ipython().run_line_magic('matplotlib', 'inline')


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = create_dataset(m)

Tx = 20
sources, targets = zip(*dataset)
sources = np.array([string_to_int(i, Tx, human_vocab) for i in sources])
targets = [string_to_int(t, Tx, machine_vocab) for t in targets]
targets = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), targets)))


index = np.random.randint(m)

print("sources shape          : ", sources.shape)
print("targets shape          : ", targets.shape)
print("Length of human_vocab  : ", len(human_vocab))
print("Length of machine_vocab: ", len(machine_vocab))

print("\n")
print("Human-readable date                 : ", dataset[index][0])
print("Machine-readable date               : ", dataset[index][1])
print("Pre-processed human-readable date   : ", sources[index])
print("Pre-processed machine-readable date : \n", targets[index])


def model_simple_nmt(human_vocab_size, machine_vocab_size, Tx = 20):
    """
    Simple Neural Machine Translation model
    
    Arguments:
    human_vocab_size -- size of the human vocabulary for dates, it will give us the size of the embedding layer
    machine_vocab_size -- size of the machine vocabulary for dates, it will give us the size of the output vector
    
    Returns:
    model -- model instance in Keras
    """
    
    ### START CODE HERE ###
    # Define the input of your model with a shape (Tx,)
    inputs = ##TODO## : Add Input layer code here
    
    # Define the embedding layer. Embedding dimension should be 64 and input_length set to Tx.
    input_embed = ##TODO## : Add Embeddings layer code here
    
    # Encode the embeddings using a bidirectional LSTM
    enc_out = ##TODO## : Add Bidirectional layer code here
    
    # Decode the encoding using an LSTM layer
    dec_out = ##TODO## : Add LSTM layer code here
    
    ### END CODE HERE ###
    
    # Apply Dense layer to every time step
    output = TimeDistributed(Dense(machine_vocab_size, activation='softmax'))(dec_out)
    
    # Create model instance 
    model = Model(input=[inputs], output=output)

    return model


# Create model
model = model_simple_nmt(len(human_vocab), len(machine_vocab), Tx)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit([sources], targets, epochs=20, batch_size=128, validation_split=0.1, verbose=2)


EXAMPLES = ['3 May 1979', '5 Apr 09', '20th February 2016', 'Wed 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '3rd of March 2001']

def run_examples(examples):
    for example in examples:
        source = string_to_int(example, Tx, human_vocab)
        prediction = model.predict(np.array([source]))
        prediction = np.argmax(prediction[0], axis = -1)
        output = int_to_string(prediction, inv_machine_vocab)
        print("source:", example)
        print("output:", ''.join(output))
    
run_examples(EXAMPLES)


##TODO## : Add your own formats and run the model on them here


def attention_3d_block(inputs):
    """
    Implement the attention block applied between two layers
    
    Argument:
    inputs -- output of the previous layer, set of hidden states
    
    Returns:
    output_attention_mul -- inputs weighted with attention probabilities
    """
    
    # Retrieve n_h and Tx from inputs' shape. Recall: inputs.shape = (m, Tx, n_h)
    Tx = int_shape(inputs)[1]
    n_h = int_shape(inputs)[2]
    
    ### START CODE HERE ###
    # Permute inputs' columns to compute "a" of shape (m, n_h, Tx)
    a = ##TODO## : Complete the Permute layer code here
    
    # Apply a Dense layer with softmax activation. It should contain Tx neurons. a.shape should still be (m, n_h, Tx).
    a = ##TODO## : Complete the Dense layer code here
    
    # Compute the mean of "a" over axis=1 (the "hidden" axis: n_h). a.shape should now be (m, Tx)
    a = ##TODO## : Complete the Lambda layer code here

    ### END CODE HERE ###
    
    # Repeat the vector "a" n_h times. "a" should now be of shape (m, n_h, Tx)
    a = RepeatVector(n_h)(a)
    
    # Permute the 2nd and the first column of a to get a probability vector of attention. a_probs.shape = (m, Tx, n_h)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    
    # Apply the attention probabilities to the "inputs" by multiplying element-wise.
    output_attention_mul = Multiply(name='attention_mul')([inputs, a_probs])
    
    return output_attention_mul


def get_data_recurrent(n, time_steps, input_dim, attention_column=None):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    if attention_column is None:
        attention_column = np.random.randint(low=0, high=input_dim)
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


np.random.seed(1)
x, y = get_data_recurrent(n = 2, time_steps = 4, input_dim = 3, attention_column=None)
print("x.shape =", x.shape)
print("x =", x)
print()
print("y.shape =", y.shape)
print("y =", y)


X, Y = get_data_recurrent(n = 10000, time_steps = 20, input_dim = 2, attention_column = 3)


def model_attention_applied_before_lstm(Tx, n_h):
    """
    Model with attention applied BEFORE the LSTM
        
    Returns:
    model -- Keras model instance
    """
    
    # Define the input of your model with a shape (Tx,)
    inputs = Input(shape=(Tx, n_h,))
    # Add the attention block
    attention_mul = attention_3d_block(inputs)
    # Pass the inputs in a LSTM layer, return the sequence of hidden states
    attention_mul = LSTM(32, return_sequences=False)(attention_mul)
    # Apply Dense layer with sigmoid activation the output should be a single number.
    output = Dense(1, activation='sigmoid')(attention_mul)
    # Create model instance 
    model = Model(input=[inputs], output=output)
    
    return model


def model_attention_applied_after_lstm(Tx, n_x):
    """
    Model with attention applied AFTER the LSTM
    
    Returns:
    model -- Keras model instance
    """
    
    # Define the input of your model with a shape (Tx,)
    inputs = Input(shape=(Tx, n_x,))
    # Pass the inputs in a LSTM layer, return the sequence of hidden states
    lstm_out = LSTM(32, return_sequences=True)(inputs)
    # Add the attention block
    attention_mul = attention_3d_block(lstm_out)
    # Flatten the output of the attention block
    attention_mul = Flatten()(attention_mul)
    # Apply Dense layer with sigmoid activation the output should be a single number.
    output = Dense(1, activation='sigmoid')(attention_mul)
    # Create model instance 
    model = Model(input=[inputs], output=output)
    
    return model


m = model_attention_applied_after_lstm(Tx = 20, n_x = 2)


m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


m.fit([X], Y, epochs=10, batch_size=16, validation_split=0.1, verbose=2)


def get_activations(model, inputs, layer_name=None):
    """
    For a given Model and inputs, find all the activations in specified layer
    If no layer then use all layers
    
    Returns:
    activations from all the layer(s)
    """
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations


# Set the input dimensions
INPUT_DIM = 2

# Set time steps
TIME_STEPS = 20

# if True, the attention vector is shared across the input_dimensions where the attention is applied.
# Set whether the attention vector is shared
SINGLE_ATTENTION_VECTOR = False

# Set the attention model in relation to LSTM
APPLY_ATTENTION_BEFORE_LSTM = False

# Set the size of the dataset
N = 300000


inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM, attention_column=3)

if APPLY_ATTENTION_BEFORE_LSTM:
    m = model_attention_applied_after_lstm(Tx = 20, n_x = 2)
else:
    m = model_attention_applied_before_lstm(Tx = 20, n_h = 2)

m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(m.summary())


m.fit([inputs_1], outputs, epochs=1, batch_size=512, validation_split=0.1, verbose=2)

attention_vectors = []
for i in range(300):
    # Generate one training example (x, y), the attention column can be on any time-step.
    testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM, attention_column=3)
    # Extract the attention vector predicted by the model "m" on the training example "x".
    attention_vector = np.mean(get_activations(m,
                                               testing_inputs_1,
                                               layer_name='attention_vec')[0], axis=2).squeeze()
    # Append the attention vector to the list of attention vectors
    assert (np.sum(attention_vector) - 1.0) < 1e-5
    attention_vectors.append(attention_vector)
# Compute the average attention on every time-step

attention_vector_final = np.mean(np.array(attention_vectors), axis=0)


# plot part.
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                     title='Attention Mechanism as '
                                                                           'a function of input'
                                                                           ' dimensions.')
plt.show()


def model_attention_nmt(human_vocab_size, machine_vocab_size, Tx = 20):
    """
    Attention Neural Machine Translation model
    
    Arguments:
    human_vocab_size -- size of the human vocabulary for dates, it will give us the size of the embedding layer
    machine_vocab_size -- size of the machine vocabulary for dates, it will give us the size of the output vector
    
    Returns:
    model -- model instance in Keras
    """

    # Define the input of your model with a shape (Tx,)
    inputs = Input(shape=(Tx,))
    
    # Define the embedding layer. This layer should be trainable and the input_length should be set to Tx.
    input_embed = Embedding(human_vocab_size, 2*32, input_length = Tx, trainable=True)(inputs)

    ### START CODE HERE ###    
    # Encode the embeddings using a bidirectional LSTM
    enc_out = ##TODO## : Add Bidirectional layer code here
    
    # Add attention
    attention = ##TODO## : Add attention block
    
    # Decode the encoding using an LSTM layer
    dec_out = ##TODO## : Add LSTM layer code here
 
    ### END CODE HERE ###
    
    # Apply Dense layer to every time steps
    output = TimeDistributed(Dense(machine_vocab_size, activation='softmax'))(dec_out)
    
    # Create model instance 
    model = Model(input=[inputs], output=output)
    
    return model


# Create model
model_att = model_attention_nmt(len(human_vocab), len(machine_vocab))

# Compile model
model_att.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model_att.fit([sources], targets, epochs=100, batch_size=128, validation_split=0.1, verbose=2)


example = "Nov 11 1998"
source = string_to_int(example, 20, human_vocab)
prediction = model_att.predict(np.array([source]))
prediction = np.argmax(prediction[0], axis = -1)
output = int_to_string(prediction, inv_machine_vocab)
print("source:", example)
print("output:", ''.join(output))


def attention_map(model, input_vocabulary, inv_output_vocabulary, text):
    """
        visualization of attention map
    """
    # encode the string
    encoded = string_to_int(text, 20, input_vocabulary)

    # get the output sequence
    prediction = model.predict(np.array([encoded]))
    predicted_text = np.argmax(prediction[0], axis=-1)
    predicted_text = int_to_string(predicted_text, inv_output_vocabulary)

    text_ = list(text)
    # get the lengths of the string
    input_length = len(text)
    output_length = predicted_text.index('<pad>') if '<pad>' in predicted_text else len(predicted_text)
    # get the activation map
    attention_vector = get_activations(model, [encoded], layer_name='attention_vec')[0].squeeze()
    activation_map = attention_vector[0:output_length, 0:input_length]
    
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)

    # add image
    i = ax.imshow(activation_map, interpolation='nearest', cmap='gray')

    # add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Probability', labelpad=2)

    # add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()

    f.show()


attention_map(model_att, human_vocab, inv_machine_vocab, example)


get_ipython().system('head /dli/data/wmt/newstest2015.tok.bpe.32000.de')


get_ipython().system('python3 OpenSeq2Seq/run.py --config_file=OpenSeq2Seq/example_configs/nmt.json --logdir=/dli/data/nmt --mode=infer --inference_out=pred.txt')


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


m = model_attention_nmt(len(human_vocab), len(machine_vocab))

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

