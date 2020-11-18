#!/usr/bin/env python
# coding: utf-8

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
    inputs = Input(shape=(Tx,))
    
    # Define the embedding layer. Embedding dimension should be 64 and input_length set to Tx.
    # Remember that the size of the embedding is a hyper-parameters!  Choose it to be whatever gives good results.
    input_embed = Embedding(human_vocab_size, 64, input_length = Tx)(inputs)
    
    # Encode the embeddings using a bidirectional LSTM
    enc_out = Bidirectional(LSTM(32, return_sequences=True))(input_embed)
    
    # Decode the encoding using an LSTM layer
    dec_out = LSTM(32, return_sequences=True)(enc_out)
    ### END CODE HERE ###
    
    # Apply Dense layer to every time step
    output = TimeDistributed(Dense(machine_vocab_size, activation='softmax'))(dec_out)
    
    # Create model instance 
    model = Model(input=[inputs], output=output)
    
    return model


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
    a = Permute((2, 1))(inputs)
    
    # Apply a Dense layer with softmax activation. It should contain Tx neurons. a.shape should still be (m, n_h, Tx).
    a = Dense(Tx, activation='softmax')(a)
    
    # Compute the mean of "a" over axis=1 (the "hidden" axis: n_h). a.shape should now be (m, Tx)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    ### END CODE HERE ###
    
    # Repeat the vector "a" n_h times. "a" should now be of shape (m, n_h, Tx)
    a = RepeatVector(n_h)(a)
    
    # Permute the 2nd and the first column of a to get a probability vector of attention. a_probs.shape = (m, Tx, n_h)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    
    # Apply the attention probabilities to the "inputs" by multiplying element-wise.
    output_attention_mul = Multiply(name='attention_mul')([inputs, a_probs])

    
    return output_attention_mul


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
    enc_out = Bidirectional(LSTM(32, return_sequences=True))(input_embed)
    
    # Add attention
    attention = attention_3d_block(enc_out)
    
    # Decode the encoding using an LSTM layer
    dec_out = LSTM(32, return_sequences=True)(attention)
 
    ### END CODE HERE ###
    
    # Apply Dense layer to every time steps
    output = TimeDistributed(Dense(machine_vocab_size, activation='softmax'))(dec_out)
    
    # Create model instance 
    model = Model(input=[inputs], output=output)
    
    return model

