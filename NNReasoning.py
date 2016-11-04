'''
    implements the work in "Towards Neural Network-based Reasoning"
    https://arxiv.org/abs/1508.05508
'''

import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell, DropoutWrapper, MultiRNNCell

main()


class MemNet():
    def __init__(self, depth_dnn, depth_rnn, num_reasoning, dropout):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        self.depth_dnn = depth_dnn
        self.depth_rnn = depth_rnn
        self.num_reasoning = num_reasoning
        self.dropout = dropout

    def generate_dnn_layer(self, encoded):
        pass

    def generate_rnn_layer(self, inputs):
        '''
            rnn layer encodes facts and qeustions
            inputs is in dimension:
                batch_size * num_sentences * word_vector_length * max_length
        '''
        batch, num_sentences, word_vector_length, max_length = inputs.shape
        cell = GRUCell(num_sentences)
        cell = DropoutWrapper(cell, output_keep_prob = self.dropout)
        encoder = MultiRNNCell([cell] * self.depth_rnn)
        pass

if __name__ == '__main__':
    main()
