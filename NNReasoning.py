'''
    implements the work in "Towards Neural Network-based Reasoning"
    https://arxiv.org/abs/1508.05508
'''

import tensorflow as tf
import numpy as np
from util import *


def main():
    vocab, train, test = input_data_parser('en', '10k', 19)
    x, y = train
    sentence_num, batch_size, sentence_length, word_vector_size = x.shape
    answer_length = len(y[0])
    net = MemNet(batch_size, 3, 3, 50, 3, 0.9)
    train, test = net.get_model(sentence_num - 1,
                                [sentence_length, word_vector_size],
                                answer_length)
    def get_word(vects):
        sentence = ""
        for vect in vects:
            sentence +=  vocab.similar_by_vector(vect, topn=1)[0][0] + ' '
        return sentence
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.summary.FileWriter('log', tf.get_default_graph())
        for epoch_idx in range(100000):
            state = sess.run(net.init_state)
            feed = {}
            for idx in range(len(train.inputs)):
                feed[train.inputs[idx]] = x[idx]

            for idx, init_state in enumerate(net.init_state):
                feed[init_state] = state[idx]

            feed[train.y] = y
            answer, loss, _ = sess.run(
                [train.anwser, train.loss, train.train_op], feed_dict=feed)
            hit = 0
            for idx, i in enumerate(answer):
                p, p_  =  np.argmax(i), np.argmax(y[idx])
                if p == p_:
                    hit += 1
            print hit * 1./idx, loss
        tf.scalar_summary('loss', loss)
        tf.scalar_summary('hit', hit)

    return

class MemNet():
    class Model:
        def __init__(self):
            pass

    def __init__(self, batch_size, depth_cnn, depth_rnn, unit_size, num_reasoning, dropout):
        self.depth_cnn = depth_cnn
        self.depth_rnn = depth_rnn
        self.num_reasoning = num_reasoning
        self.dropout = dropout
        self.unit_size = unit_size
        self.batch_size = batch_size

        # init params
        self.filter = tf.random_normal([100, 1, 1, 1], dtype = tf.float32)
        self.strides = [1, 1, 1, 1]
        self.encode_steps = 10


        self.init_state = []

    def expand_sentence_tensor(self, sentence):
        '''
            convert sentence tensor to a 4D tensor that can be conved
        '''
        # currently a sentence is a 2d vector which is batch_size * sentence_rep_length
        rlt = tf.expand_dims(sentence, 2)
        rlt = tf.expand_dims(rlt, 3)
        return rlt

    def get_model(self, fact_num, input_dimension, answer_length):
        train = self.Model()
        test = self.Model()
        with tf.name_scope('Train'):
            with tf.variable_scope('Model'):
                (train.inputs,
                 train.anwser,
                 train.y,
                 train.train_op,
                 train.loss)= self.generate_model(
                     fact_num, input_dimension, answer_length, train = True)
        with tf.name_scope('Test'):
            with tf.variable_scope('Model', reuse = True):
                test.inputs, test.anwser, _, _, test.loss = self.generate_model(
                    fact_num, input_dimension, answer_length, train = False)
        self.train = train
        self.test = test
        return self.train, self.test

    def generate_model(self, fact_num, input_dimension, answer_length, train = False):
        '''
            fact_num is the number of fact sentences
            input_dimension is:
                batch * sentence_length * word vector size
        '''
        encoder = self.generate_encoder(fact_num + 1, input_dimension, train)
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
        anwser = self.generate_decoder(question, answer_length)
        if train:
            target = tf.placeholder(tf.float32, shape = anwser.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(anwser, target)
            loss = tf.reduce_mean(loss)
            # for v in tf.trainable_variables():
            #     loss += tf.nn.l2_loss(v)
            # loss = 0.
            # labels = tf.cast(tf.argmax(target, axis=1), tf.int32)
            # l = anwser.get_shape()[0]
            # classes = anwser.get_shape()[1]
            # for i in range(l):
            #     for j in range(classes):
            #         loss += tf.maximum(0., anwser[i, j] - anwser[i, labels[i]] + 1.)
            optimizer = tf.train.GradientDescentOptimizer(1e-9)
            vs = tf.trainable_variables()
            gs = tf.gradients(loss, vs)
            gs, _ = tf.clip_by_global_norm(gs, 1)
            gvs = zip(gs, vs)
            train_op = optimizer.apply_gradients(gvs, global_step=tf.contrib.framework.get_or_create_global_step())
        else:
            train_op = tf.no_op()
            target = 1
            loss = 1

        return inputs, anwser, target, train_op, loss

    def generate_rnn_layer(self, unit_size, train):
        '''
            rnn layer for encoder layer
        '''
        cell = tf.nn.rnn_cell.GRUCell(unit_size)
        if train:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.dropout)
        layer = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * self.depth_rnn, state_is_tuple = True)
        state = layer.zero_state(self.batch_size, tf.float32)
        self.init_state.append(state)
        return state, layer

    def generate_encoder(self, num_sentence, input_shape, train):
        '''
            encoding layer to encode facts and question
            the input_shape should be the shape of:
                sentence length * word vector length
        '''
        sentence_holder = []
        init_state_holder = []
        for _ in range(num_sentence):
            if _ == num_sentence -1:
                scope_name = "question"
            else:
                scope_name = 'fact_' + str(_)
            with tf.variable_scope(scope_name) as scope:
                tmp = tf.placeholder(
                    tf.float32,
                    shape = [self.batch_size, input_shape[0], input_shape[1]])
                state, layer = self.generate_rnn_layer(self.unit_size, train)
                for step in range(input_shape[0]):
                    if step > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = layer(tmp[:, step, :], state)
                # only the take the last output as the sentence representation
                sentence_holder.append((output, tmp))
        return sentence_holder

    def generate_conv_layer(self, question, fact):
        feed = tf.concat(1, [question, fact])
        for i in range(self.depth_cnn):
            with tf.variable_scope('depth_' + str(i)) as scope:
                feed = tf.nn.conv2d(
                    feed, filter = self.filter, strides = self.strides, padding = 'SAME')
                feed = tf.nn.relu(feed)
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
        question = tf.nn.max_pool(question, [1, 5, 1, 1], [1, 5, 1, 1], padding = 'SAME')
        return (question, fs)

    def generate_decoder(self, content, options):
        '''
            content is of shape
                batch * content dimension * 1 * 1
        '''
        # convert to shape batch * sequence
        content = content[:, :, 0, 0]
        batch, content_length = content.get_shape()
        softmax_w = tf.get_variable('softmax_w', [content_length, options], dtype = tf.float32)
        softmax_b = tf.ones([options], dtype = tf.float32)
        output = tf.batch_matmul(content, softmax_w)
        output = tf.add(output, softmax_b)
        output = tf.nn.softmax(output)
        output = tf.nn.relu(output)
        return output

if __name__ == '__main__':
    main()
