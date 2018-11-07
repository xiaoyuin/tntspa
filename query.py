import argparse
import json
import os
import pathlib

from generator_utils import query_dbpedia, decode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='A file containing lines of SPARQL queries')
    parser.add_argument('--output_path', nargs='?', type=str, default='output.txt')
    parser.add_argument('--reference', nargs='?', default=None, dest='reference_path', help='For comparing the results between two query files')
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    reference_path = args.reference_path
    
    input_file = open(input_path, 'r')
    output_file = open(output_path, 'w')
    for line in input_file:
        line = str(line.strip())
        output_file.write(json.dumps(query_dbpedia(decode(line))))
        output_file.write('\n')
    input_file.close()
    output_file.close()