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
    
    # Decode the encoding using an LSTM layer
    dec_out = LSTM(32, name='Decoder', return_sequences=True)(enc_out)
    
    ### END CODE HERE ###
    
    # Apply Dense layer to every time step
    output = TimeDistributed(Dense(machine_vocab_size, activation='softmax'))(dec_out)
    
    # Create model instance 
    model = Model(input=[inputs], output=output)

    return model