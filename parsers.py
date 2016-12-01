'''
    these data parsing funcitons should return:
        word dictionaries
        train and test data in following format:
            a dictionary of tensors of shape [batch * sentence_length * word_vector_length]
                each tensor represents a factor, the last tensor is the question
                the key is their sequence
            a list of results
'''

import numpy as np
import gensim

word_vector_dimension = 10
sentence_stop = np.ones(word_vector_dimension) * -1
place_holder = np.zeros(word_vector_dimension)

def tokenize(model, data, label):
    x = {}
    y = []
    max_length = 0
    # convert sentences to vectors
    for i in range(len(label)):
        for idx , s in enumerate(data[i]):
            fact = []
            for word in s.split():
                fact.append(model[word])
            fact.append(sentence_stop)
            if idx in x:
                x[idx].append(fact)
            else:
                x[idx] = [fact]
            if len(fact) > max_length:
                max_length = len(fact)
        answer = []
        for word in label[i].split():
            answer.append(model[word])
        answer.append(sentence_stop)
        y.append(answer)

    # pending shorter sentences with trailing 0s
    for i in x:
        facts = x[i]
        for idx in range(len(facts)):
            fact = facts[idx]
            if len(fact) < max_length:
                facts[idx] = fact + [place_holder for _ in range(max_length - len(fact))]
    rlt = []
    for i in sorted(x.keys()):
        rlt.append(x[i])
    x = np.asarray(rlt)
    y = np.asarray(y)
    return x, y

def vocabulary(sentences):
    model = gensim.models.Word2Vec(sentences,
                                   size=10,
                                   window=5,
                                   min_count=5,
                                   workers=4)
    return model

def parse_single_supporting_fact():
     raise NotImplementedError("parse_single_supporting_fact is not implemented yet")


def parse_two_supporting_facts():
     raise NotImplementedError("parse_two_supporting_facts is not implemented yet")


def parse_three_supporting_facts():
     raise NotImplementedError("parse_three_supporting_facts is not implemented yet")


def parse_two_arg_relations():
     raise NotImplementedError("parse_two_arg_relations is not implemented yet")


def parse_three_arg_relations():
     raise NotImplementedError("parse_three_arg_relations is not implemented yet")


def parse_yes_no_questions():
     raise NotImplementedError("parse_yes_no_questions is not implemented yet")


def parse_counting():
     raise NotImplementedError("parse_counting is not implemented yet")


def parse_lists_sets():
     raise NotImplementedError("parse_lists_sets is not implemented yet")


def parse_simple_negation():
     raise NotImplementedError("parse_simple_negation is not implemented yet")


def parse_indefinite_knowledge():
     raise NotImplementedError("parse_indefinite_knowledge is not implemented yet")


def parse_basic_coreference():
     raise NotImplementedError("parse_basic_coreference is not implemented yet")


def parse_conjunction():
     raise NotImplementedError("parse_conjunction is not implemented yet")


def parse_compound_coreference():
     raise NotImplementedError("parse_compound_coreference is not implemented yet")


def parse_time_reasoning():
     raise NotImplementedError("parse_time_reasoning is not implemented yet")


def parse_basic_deduction():
     raise NotImplementedError("parse_basic_deduction is not implemented yet")


def parse_basic_induction():
     raise NotImplementedError("parse_basic_induction is not implemented yet")


def parse_positional_reasoning():
     raise NotImplementedError("parse_positional_reasoning is not implemented yet")


def parse_size_reasoning():
     raise NotImplementedError("parse_size_reasoning is not implemented yet")


def parse_path_finding(path):
    train_file = path + '/' + 'qa19_path-finding_train.txt'
    test_file = path + '/' + 'qa19_path-finding_test.txt'

    train_data = []
    train_y = []
    test_data = []
    test_y = []
    all_sentences = []
    tmp = []
    with open(train_file) as f:
        for l in f:
            l = l.strip().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ')
            all_sentences.append(l.split())
            if l.startswith('6'):
                qeustion, answer, _ = l.split('\t')
                tmp.append(qeustion)
                train_data.append(tmp)
                train_y.append(answer)
                tmp = []
            else:
                tmp.append(l)

    with open(test_file) as f:
        for l in f:
            l = l.strip().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ')
            all_sentences.append(l.split())
            if l.startswith('6'):
                question, answer, _ = l.split('\t')
                tmp.append(qeustion)
                test_data.append(tmp)
                test_y.append(answer)
                tmp = []
            else:
                tmp.append(l)

    model = vocabulary(all_sentences)

    def discretizing_answer(y):
        rlts = []
        y_ = []
        for i in y:
            seen = False
            for idx, rlt in enumerate(rlts):
                if np.array_equal(rlt, i):
                    seen = True
                    y_.append(idx)
            if not seen:
                y_.append(len(rlts))
                rlts.append(i)
        y = [[1 if j==i else 0 for j in range(len(rlts))] for i in y_]
        return np.asarray(y)

    x, y = tokenize(model, train_data, train_y)
    y = discretizing_answer(y)
    train = (x, y)
    x, y = tokenize(model, test_data, test_y)
    y = discretizing_answer(y)
    test = (x, y)

    return model, train, test

def parse_agents_motivations():
     raise NotImplementedError("parse_agents_motivations is not implemented yet")


