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
    x, y = train
    sentence_num, batch_size, sentence_length, word_vector_size = x.shape
    print net.get_model(sentence_num - 1, [batch_size, sentence_length, word_vector_size])
    return

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
                self.filter = tf.random_normal([3, 1, 1, 1], dtype = tf.float64)
        self.strides = [1, 1, 1, 1]

    def expand_sentence_tensor(self, sentence):
        '''
            convert sentence tensor to a 4D tensor that can be conved
        '''
        # currently a sentence is a 2d vector which is batch_size * sentence_rep_length
        rlt = tf.expand_dims(sentence, 2)
        rlt = tf.expand_dims(rlt, 3)
        return rlt

    def get_model(self, fact_num, input_dimension):
        with self.sess.as_default():
            with self.graph.as_default():
                return self.generate_model(fact_num, input_dimension)

    def generate_model(self, fact_num, input_dimension):
        '''
            fact_num is the number of fact sentences
            input_dimension is:
                batch * sentence_length * word vector size
        '''
        encoder = self.generate_encoder(fact_num + 1, input_dimension)
        inputs = []
        facts = []
        for idx, encodes in enumerate(encoder):
            output, feed = encodes
            inputs.append(feed)
            output = self.expand_sentence_tensor(output)
            if idx == fact_num:
                # question is always the last tensor
                question = output
            else:
                facts.append(output)

        for layer in range(self.num_reasoning):
            with tf.variable_scope('reasoning_layer_' + str(layer)) as scope:
                question, facts = self.generate_resaoner(question, facts)
        answer = self.generate_decoder(question, input_dimension[1])
        return inputs, answer

    def generate_rnn_layer(self, unit_size):
        '''
            rnn layer for encoder layer
        '''
        cell = tf.nn.rnn_cell.GRUCell(unit_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.dropout)
        layer = tf.nn.rnn_cell.MultiRNNCell([cell] * self.depth_rnn)
        return layer

    def generate_encoder(self, num_sentence, input_shape):
        '''
            encoding layer to encode facts and question
            the input_shape should be the shape of:
                batch * sentence length * word vector length
        '''
        sentence_holder = []
        for _ in range(num_sentence):
            if _ == num_sentence -1:
                scope_name = "question"
            else:
                scope_name = 'fact_' + str(_)
            with tf.variable_scope(scope_name) as scope:
                tmp = tf.placeholder(tf.float64, shape = input_shape)
                layer = self.generate_rnn_layer(self.unit_size)
                # TODO: adding sequence length here and compare
                output, last_state = tf.nn.dynamic_rnn(
                                            cell = layer,
                                            dtype = tf.float64,
                                            inputs = tmp
                                            )
                # only the take the last output as the sentence representation
                sentence_holder.append((output[:, -1, :], tmp))
        return sentence_holder

    def generate_conv_layer(self, question, fact):
        feed = tf.concat(1, [question, fact])
        for i in range(self.depth_cnn):
            with tf.variable_scope('depth_' + str(i)) as scope:
                feed = tf.nn.conv2d(feed, filter = self.filter, strides = self.strides, padding = 'SAME')
        return tf.split(1, 2, feed)

    def generate_resaoner(self, question, facts):
        qs = []
        fs = []
        for idx, fact in enumerate(facts):
            with tf.variable_scope('sentence_' + str(idx)) as scope:
                q, f = self.generate_conv_layer(question, fact)
                qs.append(q)
                fs.append(f)
        question = tf.concat(1, qs)
        question = tf.nn.avg_pool(question, [1, 5, 1, 1], [1, 5, 1, 1], padding = 'SAME')
        return (question, fs)

    def generate_decoder(self, content, sentence_length):
        '''
            content is of shape
                batch * content dimension * 1 * 1
        '''
        # convert to shape batch * sequence * content dimension
        content = tf.expand_dims(content[:, :, 0, 0], 1)
        # append null inputs until it reaches sentence length limit
        batch, _, content_length =  content.get_shape()
        content = tf.concat(1, [content, tf.zeros([batch, sentence_length - 1, content_length], dtype = tf.float64)])
        # feed in rnn to get a sentence
        layer = self.generate_rnn_layer(1)
        # TODO: adding seqeunce length here and compare
        output, last_state = tf.nn.dynamic_rnn(cell = layer,
                                               dtype = tf.float64,
                                               inputs = content)
        return output

if __name__ == '__main__':
    main()
