'''
    implements the work in "Towards Neural Network-based Reasoning"
    https://arxiv.org/abs/1508.05508
'''

import tensorflow as tf
from util import *


def main():
    pass

class MemNet():
    def __init__(self, depth_dnn, depth_rnn, unit_size, num_reasoning, dropout):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        self.depth_dnn = depth_dnn
        self.depth_rnn = depth_rnn
        self.num_reasoning = num_reasoning
        self.dropout = dropout
        self.unit_size = unit_size

    def generate_dnn_layer(self, encoded):
        pass

    def generate_rnn_layer(self, batch, num_sentences, word_vector_length, max_length):
        '''
            rnn layer encodes facts and qeustions
            inputs is in dimension:
                batch_size * num_sentences * word_vector_length * max_length
        '''
        cell = tf.nn.rnn_cell.GRUCell(self.unit_size)
        cell = DropoutWrapper(cell, output_keep_prob = self.dropout)
        encoder = MultiRNNCell([cell] * self.depth_rnn)
        pass

if __name__ == '__main__':
    main()
