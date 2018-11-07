import os
import argparse
import pandas as pd
import numpy as np
import json
import moses
from tqdm import tqdm
from generator_utils import query_dbpedia, decode
from bleu import compute_bleu


def compute_query_accuracy(res_reference, res_translation):
    """Compare two results from executing the SPARQL queries, and return the query accuracy"""
    
    return 1 if res_reference == res_translation else 0

parser = argparse.ArgumentParser()
parser.add_argument('data_directory')
parser.add_argument('result_diretory')
args = parser.parse_args()

data_directory = args.data_directory
result_diretory = args.result_diretory

dev_source_path = os.path.join(data_directory, 'dev.en')
dev_reference_path = os.path.join(data_directory, 'dev.sparql')
dev_translation_path = os.path.join(result_diretory, 'dev_translation.sparql')
test_source_path = os.path.join(data_directory, 'test.en')
test_reference_path = os.path.join(data_directory, 'test.sparql')
test_translation_path = os.path.join(result_diretory, 'test_translation.sparql')

dev_sources = []
dev_references = []
dev_translations = []
test_sources = []
test_references = []
test_translations = []

with open(dev_source_path) as f:
    dev_sources = f.readlines()
with open(dev_reference_path) as f:
    dev_references = f.readlines()
with open(dev_translation_path) as f:
    dev_translations = f.readlines()
with open(test_source_path) as f:
    test_sources = f.readlines()
with open(test_reference_path) as f:
    test_references = f.readlines()
with open(test_translation_path) as f:
    test_translations = f.readlines()

# Write dev results and statistics
dev_results = []
source_lengths = []
target_lengths = []
translation_lengths = []
bleus = []
query_accuracys = []

tokenizer = moses.MosesTokenizer()
for s, r, t in zip(tqdm(dev_sources, desc="Generating results on dev set"), dev_references, dev_translations):
    # fetch results and record
    s = s.strip()
    r = r.strip()
    t = t.strip()
    result = {"reference": r, "translation": t}
    result["reference_result"] = query_dbpedia(decode(r))
    result["translation_result"] = query_dbpedia(decode(t))
    dev_results.append(result)
    # write statistics
    s_tokens = tokenizer.tokenize(s)
    r_tokens = r.split()
    t_tokens = t.split()
    source_lengths.append(len(s_tokens))
    target_lengths.append(len(r_tokens))
    translation_lengths.append(len(t_tokens))
    bleu_tuple = compute_bleu([[r]], [t])
    bleus.append(bleu_tuple[0])
    query_accuracys.append(compute_query_accuracy(result["reference_result"], result["translation_result"]))

with open(os.path.join(result_diretory, 'dev_results.json'), 'w') as f:
    f.write(json.dumps(dev_results))

dev_statistics = pd.DataFrame(data={"source_length":source_lengths, "target_length":target_lengths, "translation_length":translation_lengths, "bleu":bleus, "query_accuracy":query_accuracys})
dev_statistics.to_csv(os.path.join(result_diretory, 'dev_statistics.csv'))

test_results = []
source_lengths.clear()
target_lengths.clear()
translation_lengths.clear()
bleus.clear()
query_accuracys.clear()

# Write test results and statistics
for s, r, t in zip(tqdm(test_sources, desc="Generating results on test set"), test_references, test_translations):
    # fetch results and record
    s = s.strip()
    r = r.strip()
    t = t.strip()
    result = {"reference": r, "translation": t}
    result["decoded_reference"] = decode(r)
    result["decoded_translation"] = decode(t)
    result["reference_result"] = query_dbpedia(result["decoded_reference"])
    result["translation_result"] = query_dbpedia(result["decoded_translation"])
    test_results.append(result)
    # write statistics
    s_tokens = tokenizer.tokenize(s)
    r_tokens = r.split()
    t_tokens = t.split()
    source_lengths.append(len(s_tokens))
    target_lengths.append(len(r_tokens))
    translation_lengths.append(len(t_tokens))
    bleu_tuple = compute_bleu([[r]], [t])
    bleus.append(bleu_tuple[0])
    query_accuracys.append(compute_query_accuracy(result["reference_result"], result["translation_result"]))

with open(os.path.join(result_diretory, 'test_results.json'), 'w') as f:
    f.write(json.dumps(test_results))

test_statistics = pd.DataFrame(data={"source_length":source_lengths, "target_length":target_lengths, "translation_length":translation_lengths, "bleu":bleus, "query_accuracy":query_accuracys})
test_statistics.to_csv(os.path.join(result_diretory, 'test_statistics.csv'))