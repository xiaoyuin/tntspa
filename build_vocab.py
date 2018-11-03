#!/usr/bin/env python
"""
Usage: python3 build_vocab.py data.en > vocab.en
"""
# import numpy as np
# from tensorflow.contrib import learn
import sys
import argparse
import nltk
import moses

def build_vocabulary_single_file(file_path):

    lines = list()
    with open(file_path) as f:
        for line in f:
            lines.append(str(line[:-1]))

    vocabulary = set()
    dictionary = dict()

    lang = file_path.split('.')[-1].lower()

    if lang == "sparql":
        
        for line in lines:
            for token in line.split(" "):
                vocabulary.add(token)
                if token in dictionary:
                    dictionary[token] += 1
                else:
                    dictionary[token] = 1

    else: # any other language

        m = moses.MosesTokenizer()
        for line in lines:
            for token in m.tokenize(line):
                vocabulary.add(token)
                if token in dictionary:
                    dictionary[token] += 1
                else:
                    dictionary[token] = 1

        # # lines = ['This is a cat','This must be boy', 'This is a a dog']
        # max_document_length = max([len(x.split(" ")) for x in lines])

        # ## Create the vocabularyprocessor object, setting the max lengh of the documents.
        # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

        # ## Transform the documents using the vocabulary.
        # x = np.array(list(vocab_processor.fit_transform(lines)))    

        # ## Extract word:id mapping from the object.
        # vocab_dict = vocab_processor.vocabulary_._mapping

        # ## Sort the vocabulary dictionary on the basis of values(id).
        # ## Both statements perform same task.
        # #sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
        # sorted_vocab = sorted(list(vocab_dict.items()), key = lambda x : x[1])

        # ## Treat the id's as index into list and create a list of words in the ascending order of id's
        # ## word with id i goes at index i of the list.
        # vocabulary = set(list(zip(*sorted_vocab))[0])
        
        # # split also by apostrophe
        # to_remove = set()
        # to_add = set()
        # for t0 in vocabulary:
        #     if "'" in t0:
        #         to_remove.add(t0)
        #         for t1 in t0.split("'"):
        #             to_add.add(t1)
        # for t0 in to_remove:
        #     vocabulary.remove(t0)
        # for t0 in to_add:
        #     vocabulary.add(t0)
        
    return vocabulary


parser = argparse.ArgumentParser()
parser.add_argument("file_paths", nargs='+')
args = parser.parse_args()

result = set()
for file_path in args.file_paths:
    result.update(build_vocabulary_single_file(file_path))
if "" in result:
    result.remove("")
auxiliary_tokens = ["<unk>", "<s>", "</s>"]
for t in auxiliary_tokens:
    if t in result:
        result.remove(t)
result = auxiliary_tokens + list(result)
result
for v in result:
    print(v)