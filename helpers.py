import tensorflow as tf
import io
import pandas as pd
import csv
import json
import math
import matplotlib
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import preprocess_sentence

def load_vectors(fname):
    """
    Load word embeddings from a pre-trained vector file using local python 
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # number of words and dimensions
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return n, d, data

def read_qald_json(file_path, lang='en', written_to_file=True):
    """
    Read json file from QALD challenges and return a list of question (given language) to sparql query pairs
    """
    with open(file_path, 'r') as json_file:
        # json_object should have one dataset field and a questions array field,
        # each question contains id, answertype, aggregation, onlydbo, hybrid, question, query, answers, etc.
        json_object = json.load(json_file)
        dataset_id = json_object["dataset"]["id"]
        questions = json_object["questions"]
        result = []
        for item in questions:
            query_sparql = item["query"]["sparql"]
            for question in item["question"]:
                if question["language"] == lang:
                    result.append((question["string"], query_sparql))
                    break
        data = pd.DataFrame(data=result, columns=["question", "sparql"])

        if written_to_file:
            base_file_path = file_path[0:file_path.rfind('.')]
            question_file = open(base_file_path+'.'+lang, 'w')
            sparql_file = open(base_file_path+'.sparql', 'w')
            for i in range(len(data)):
                question_file.write(data["question"][i])
                question_file.write('\n')
                sparql_file.write(data["sparql"][i])
                sparql_file.write('\n')
            question_file.close()
            sparql_file.close()

        return data

def read_qald_csv(file_path, written_to_file=True):
    """
    Read csv file from QALD challenges and return a pandas DataFrame containing its data
    """
    data = pd.read_csv(file_path, sep=';', names=["type", "question", "sparql", "other1", "other2"])

    if written_to_file:
            base_file_path = file_path[0:file_path.rfind('.')]
            question_file = open(base_file_path+'.en', 'w')
            sparql_file = open(base_file_path+'.sparql', 'w')
            for i in range(len(data)):
                question_file.write(data["question"][i])
                question_file.write('\n')
                sparql_file.write(data["sparql"][i])
                sparql_file.write('\n')
            question_file.close()
            sparql_file.close()

    return data

def tokenize(sentence):
    """Split the given sentence into a list of tokens"""
    return sentence.split('')

def build_vocabulary(file_path):
    """
    Build a vocabulary from given file path, and output a vocabulary file?
    """
    vocabulary = set()
    file = open(file_path)

    # Simple version, split lines into words using space as delimiter
    # maybe works, maybe not because many words in sparql are combined like functions
    for line in file:
        for word in line.split(' '):
            vocabulary.add(word)

    return vocabulary

def read_dbpedia_prefix(file_path="data/prefixes.csv"):
    """
    Read prefixes from DBpedia
    """
    result = {}
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            result[row[0]] = row[1]

    # Or we can use pandas to read csv into a DataFrame object
    # data = pd.read_csv("data/prefixes.csv", sep='\t')
    # return data

    return result

def read_lines(file_path):
    input_texts = []
    with open(file_path) as file:
        for line in file:
            input_texts.append(line)
    return input_texts

def build_vocabulary_from_texts(texts):
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    return vocab

def write_history(history, file_location):
    file = open(file_location, 'w')
    json.dump({
        "history": history.history,
        "params": history.params,
        "epoch": history.epoch
    }, file)

def read_history(file_location):
    file = open(file_location, 'r')
    history = json.load(file)
    return history

def plot_ppls(train_ppl, dev_ppl, test_ppl, step_or_epoch, save_paths):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    x_label = 'Step' if step_or_epoch else 'Epoch'
    x_col = 'Step' if step_or_epoch else 'epoch'
    y_col_train = 'Value' if step_or_epoch else 'ppl'
    y_col_valid = 'Value' if step_or_epoch else 'valid_ppl'
    y_col_test = 'Value' if step_or_epoch else 'test_ppl'

    train_ppl.plot(x=x_col, y=y_col_train, ax=ax, label='Train')
    dev_ppl.plot(x=x_col, y=y_col_valid, ax=ax, label='Valid')
    if test_ppl is not None:
        test_ppl.plot(x=x_col, y=y_col_test, ax=ax, label='Test')

    ylim = max(train_ppl.iloc[-1][y_col_train], dev_ppl.iloc[-1][y_col_valid])
    if test_ppl is not None:
        ylim = max(ylim, test_ppl.iloc[-1][y_col_test])
    ylim = math.ceil(ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Perplexity')
    ax.set_ylim(0, ylim)
    ax.grid(True)
    
    for save_path in save_paths:
        fig.savefig(save_path, dpi=150)

    plt.close(fig=fig)

def read_fairseq_history(file_path):

    def read_line_into_dict(line):
        d = {}
        cols = [ c.strip() for c in line.split('|') ]
        for col in cols:
           parts = col.split()
           if len(parts) == 2:
               d[parts[0]] = float(parts[1]) if parts[1].isnumeric() else parts[1]
        return d

    train_ppl = pd.DataFrame()
    valid_ppl = pd.DataFrame()
    test_ppl = pd.DataFrame()

    with open(file_path) as f:
        for line in f.readlines():
            if line.startswith('| epoch'):
                line = line[1:-1]
                d = read_line_into_dict(line)
                if "valid on 'valid' subset" in line:
                    valid_ppl = valid_ppl.append(d, ignore_index=True)
                elif "valid on 'test' subset" in line:
                    test_ppl = test_ppl.append(d, ignore_index=True)
                else:
                    train_ppl = train_ppl.append(d, ignore_index=True)
    
    return train_ppl, valid_ppl, test_ppl