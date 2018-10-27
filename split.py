import argparse
import random
import os

TRAINING_PERCENTAGE = 80
TEST_PERCENTAGE = 10
DEV_PERCENTAGE = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_rates', dest='split', default='80/10/10', help='how to split the dataset')
    parser.add_argument('data_dir')
    parser.add_argument('--data_prefix', default='data')
    args = parser.parse_args()
    
    split = args.split
    data_prefix = args.data_prefix
    data_dir = args.data_dir
    sparql_file = os.path.join(data_dir, data_prefix + '.sparql')
    en_file = os.path.join(data_dir, data_prefix + '.en')

    random.seed()
    lines = -1
    with open(en_file) as f:
        lines = len(f.readlines())
    
    if split is not None:
        splits = list(map(int, split.split('/')))
        TRAINING_PERCENTAGE = splits[0]
        DEV_PERCENTAGE = splits[1]
        TEST_PERCENTAGE = splits[2]
    
    split1 = int(TRAINING_PERCENTAGE / 100 * lines)
    split2 = int((TRAINING_PERCENTAGE+DEV_PERCENTAGE) / 100 * lines )

    indexes = list(range(lines))
    random.shuffle(indexes)

    with open(sparql_file) as original_sparql, open(en_file) as original_en:
        sparql = original_sparql.readlines()
        english = original_en.readlines()

        with open(os.path.join(data_dir, 'train.sparql'), 'w') as train_sparql, open(os.path.join(data_dir, 'train.en'), 'w') as train_en, open(os.path.join(data_dir, 'dev.sparql'), 'w') as dev_sparql, open(os.path.join(data_dir, 'dev.en'), 'w') as dev_en, open(os.path.join(data_dir, 'test.sparql'), 'w') as test_sparql, open(os.path.join(data_dir, 'test.en'), 'w') as test_en:
            for i in indexes[:split1]:
                train_sparql.write(sparql[i])
                train_en.write(english[i])
            for i in indexes[split1:split2]:
                dev_sparql.write(sparql[i])
                dev_en.write(english[i])
            for i in indexes[split2:]:
                test_sparql.write(sparql[i])
                test_en.write(english[i])
