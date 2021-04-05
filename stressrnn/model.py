#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O., MashaPo

'''
BiLSTM model for predicting the stress position in a word.
'''

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Activation, Sequential


class LSTMStressModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(LSTMStressModel, self).__init__()
        self.bidirectional_1 = Bidirectional(LSTM(64), input_shape=input_shape)
        self.dropout_1 = Dropout(0.2)
        self.dense_1 = Dense(40)
        self.activation_1 = Activation(activation='softmax')


    def call(self, inputs):
        output = self.bidirectional_1(inputs)
        output = self.dense(self.dropout_1(output))
        return self.activation_1(output)