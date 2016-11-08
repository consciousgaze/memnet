'''
    these data parsing funcitons should return:
        word dictionaries
        train and test data in word vector
'''

#TODO: use vector representation for word instead of simple indexing
def tokenize(w2i, data, label):
    x = []
    y = []
    for i in range(len(label)):
        facts = []
        for s in data[i]:
            fact = []
            for word in s.split():
                fact.append(w2i[word])
            facts.append(fact)
        x.append(facts)
        answer = []
        for word in label[i].split():
            answer.append(w2i[word])
        y.append(answer)
    return x, y

def vocabulary(text):
    words = set(text.split())
    word_to_idx = {}
    idx_to_word = {}
    for idx, word in enumerate(words):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
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


