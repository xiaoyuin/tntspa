import io
import tensorflow as tf
import pandas as pd
import json
import fastText
import csv
import re

def load_vectors(fname):
    """
    Load word embeddings from a pre-trained vector file
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # number of words and dimensions
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

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

def preprocess_sentence(w):
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w

def preprocess_sparql(s):
    """
    Add start symbol and end symbol to a sparql query for decoder processing
    """

    s = re.sub(r"PREFIX\s[^\s]*\s[^\s]*", "", s)

    return "<s> "+s.rstrip().strip()+" </s>" 