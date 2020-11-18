#!/usr/bin/env python
# coding: utf-8

inputs = Input((4,))

embedding_layer = Embedding(num_classes, EMBEDDING_SIZE, input_length=2*window_size, name='embedding_layer')
mean_layer = Lambda(lambda x: K.mean(x, axis=1))
output_layer = Dense(num_classes, activation='softmax')


output = embedding_layer(inputs)
output = mean_layer(output)
output = output_layer(output)

model = Model(inputs=[inputs], outputs=output)

optimizer = RMSprop(lr=0.1, rho=0.99)

