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
    g = net.graph
    s = net.sess
    fact = net.process_data(train[0])
    with s.as_default():
        with g.as_default():
            encoder = net.generate_encoder(6, np.asarray(fact[0]).shape)
            results = []
            for idx, encodes in enumerate(encoder):
                outputs, last_states, inputs = encodes
                result = tf.contrib.learn.run_n({"outputs": outputs, "last_states": last_states},
                                                n=1, feed_dict = {inputs: fact[idx]})
                results.append(result)
            # adding batch dimension
            question = tf.constant(np.asarray([results[-1][-1]['outputs']]))
            facts = []
            for result in results[:-1]:
                facts.append(tf.constant(np.asarray([result[-1]['outputs']])))
            reasoner = net.generate_resaoner(question, facts)
            print reasoner

class MemNet():
    def __init__(self, depth_cnn, depth_rnn, unit_size, num_reasoning, dropout):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        self.depth_cnn = depth_cnn
        self.depth_rnn = depth_rnn
        self.num_reasoning = num_reasoning
        self.dropout = dropout
        self.unit_size = unit_size

        # init params
        with self.sess.as_default():
            with self.graph.as_default():
                self.filter = tf.random_normal([1, 1, 20, 20], dtype = tf.float64)
        self.strides = [1, 2, 2, 1]


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

    def generate_conv_layer(self, question, fact):
        feed = tf.concat(0, [question, fact])
        for i in range(self.depth_cnn):
            with tf.variable_scope('depth_' + str(i)) as scope:
                print feed.get_shape(), self.filter, self.strides
                feed = tf.nn.conv2d(feed, filter = self.filter, strides = self.strides, padding = 'VALID')
        return tf.split(0, 2, feed)

    def generate_resaoner(self, question, facts):
        qs = []
        fs = []
        for idx, fact in enumerate(facts):
            with tf.variable_scope('fact_' + str(idx)) as scope:
                q, f = self.generate_conv_layer(question, fact)
                qs.append(q)
                fs.append(f)
        question = tf.concat(0, qs)
        return (question, fs)

if __name__ == '__main__':
    main()
