import os
from parsers import *

data_base = 'data/tasks_1-20_v1-2/'
word_vector_length = word_vector_dimension

task_dict = {
              1: "single-supporting-fact",
              2: "two-supporting-facts",
              3: "three-supporting-facts",
              4: "two-arg-relations",
              5: "three-arg-relations",
              6: "yes-no-questions",
              7: "counting",
              8: "lists-sets",
              9: "simple-negation",
             10: "indefinite-knowledge",
             11: "basic-coreference",
             12: "conjunction",
             13: "compound-coreference",
             14: "time-reasoning",
             15: "basic-deduction",
             16: "basic-induction",
             17: "positional-reasoning",
             18: "size-reasoning",
             19: "path-finding",
             20: "agents-motivations"
            }

parse = {
         1: parse_single_supporting_fact,
         2: parse_two_supporting_facts,
         3: parse_three_supporting_facts,
         4: parse_two_arg_relations,
         5: parse_three_arg_relations,
         6: parse_yes_no_questions,
         7: parse_counting,
         8: parse_lists_sets,
         9: parse_simple_negation,
         10: parse_indefinite_knowledge,
         11: parse_basic_coreference,
         12: parse_conjunction,
         13: parse_compound_coreference,
         14: parse_time_reasoning,
         15: parse_basic_deduction,
         16: parse_basic_induction,
         17: parse_positional_reasoning,
         18: parse_size_reasoning,
         19: parse_path_finding,
         20: parse_agents_motivations
        }

def input_data_parser(lang, size, task):
    if lang != 'en' and lang != "hn" and lang != "shuffled":
        raise Exception("Task language must be en, hn or shuffuled.")
    if size != '1k' and size != '10k' and size != None:
        raise Exception("Data size must be 1k or 10k. None specified is considered 10k.")
    if size == '10k':
        data_path = data_base + lang + '-10k'
    else:
        data_path = data_base + lang

    if task in task_dict:
        return parse[task](data_path)

    tasks = ''
    for i in task_dict:
        tasks += '\t\t' + task_dict[i] + '\n'
        if task_dict[i] == task:
            return parse[i](data_path)

    raise Exception('Task not found. Only following tasks are supported:\n' + tasks)
    pass

