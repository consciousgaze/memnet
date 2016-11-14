'''
    implements the work in "Towards Neural Network-based Reasoning"
    https://arxiv.org/abs/1508.05508
'''

import tensorflow as tf
import numpy as np
from util import *


def main():
    net = MemNet(10, 4, 20, 10, 0.3)
    w2i, i2w, train, test = input_data_parser('en', '1k', 19)
    g = tf.Graph()
    s = tf.Session(graph = g)
    fact = net.process_data(train[0])
    with s.as_default():
        with g.as_default():
            encoder = net.generate_encoder(6, np.asarray(fact[0]).shape)
            for idx, encodes in enumerate(encoder):
                outputs, last_states, inputs = encodes
                result = tf.contrib.learn.run_n({"outputs": outputs, "last_states": last_states},
                                                n=1, feed_dict = {inputs: fact[idx]})

class MemNet():
    def __init__(self, depth_dnn, depth_rnn, unit_size, num_reasoning, dropout):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        self.depth_dnn = depth_dnn
        self.depth_rnn = depth_rnn
        self.num_reasoning = num_reasoning
        self.dropout = dropout
        self.unit_size = unit_size

    def process_data(self, train):
        f1 = []
        f2 = []
        f3 = []
        f4 = []
        f5 = []
        q = []
        for d in train:
            f1.append(d[0][:9])
            f2.append(d[1][:9])
            f3.append(d[2][:9])
            f4.append(d[3][:9])
            f5.append(d[4][:9])
            q.append(d[5][:9])
        return f1, f2, f3, f4, f5, q

    def generate_dnn_layer(self, encoded):
        pass

    def generate_rnn_layer(self):
        '''
            rnn layer for encoder layer
        '''
        cell = tf.nn.rnn_cell.GRUCell(self.unit_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.dropout)
        layer = tf.nn.rnn_cell.MultiRNNCell([cell] * self.depth_rnn)
        return layer

    def generate_encoder(self, num_sentence, input_shape):
        '''
            encoding layer to encode facts and question
        '''
        sentence_holder = []
        for _ in range(num_sentence):
            with tf.variable_scope("fact_" + str(_)) as scope:
                tmp = tf.placeholder(tf.float64, shape = input_shape)
                layer = self.generate_rnn_layer()
                output, last_state = tf.nn.dynamic_rnn(
                                            cell = layer,
                                            dtype = tf.float64,
                                            inputs = tmp
                                            )
                sentence_holder.append((output, last_state, tmp))
        return sentence_holder

if __name__ == '__main__':
    main()
