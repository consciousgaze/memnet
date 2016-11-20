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

#TODO: use vector representation for word instead of simple indexing
def tokenize(w2i, data, label):
    x = {}
    y = []
    max_length = 0
    # convert sentences to vectors
    for i in range(len(label)):
        for idx , s in enumerate(data[i]):
            fact = []
            for word in s.split():
                fact.append(w2i[word])
            fact.append(-1)
            if idx in x:
                x[idx].append(fact)
            else:
                x[idx] = [fact]
            if len(fact) > max_length:
                max_length = len(fact)
        answer = []
        for word in label[i].split():
            answer.append(w2i[word])
        answer.append(-1)
        y.append(answer)

    # pending shorter sentences with trailing 0s
    for i in x:
        facts = x[i]
        for idx in range(len(facts)):
            fact = facts[idx]
            if len(fact) < max_length:
                facts[idx] = fact + [0] * (max_length - len(fact))
    rlt = []
    for i in sorted(x.keys()):
        rlt.append(x[i])

    # numpy will remove the word_vect dimension since it is of length 1
    # pending the dimension back
    x = np.expand_dims(np.asarray(rlt), 3)
    y = np.expand_dims(np.asarray(y), 2)
    return x, y

def vocabulary(text):
    words = set(text.split())
    word_to_idx = {}
    idx_to_word = {}
    for idx, word in enumerate(words):
        word_to_idx[word] = idx + 1
        idx_to_word[idx + 1] = word
    word_to_idx[None] = -1
    idx_to_word[-1] = None
    return word_to_idx, idx_to_word

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
    all_text = ""
    tmp = []
    with open(train_file) as f:
        for l in f:
            l = l.strip().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ')
            all_text += l + ' '
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
            all_text += l + ' '
            if l.startswith('6'):
                question, answer, _ = l.split('\t')
                tmp.append(qeustion)
                test_data.append(tmp)
                test_y.append(answer)
                tmp = []
            else:
                tmp.append(l)

    word_to_idx, idx_to_word = vocabulary(all_text)

    x, y = tokenize(word_to_idx, train_data, train_y)
    train = (x, y)
    x, y = tokenize(word_to_idx, test_data, test_y)
    test = (x, y)

    return word_to_idx, idx_to_word, train, test

def parse_agents_motivations():
     raise NotImplementedError("parse_agents_motivations is not implemented yet")


