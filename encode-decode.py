
import argparse
from generator_utils import decode, encode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['encode', 'decode'], default='decode')
    parser.add_argument('input_path')
    args = parser.parse_args()

    with open(args.input_path, 'r') as input_file:
        for line in input_file:
            if args.mode == 'decode':
                print(decode(line.strip()))
            elif args.mode == 'encode':
                print(encode(line.strip()))

