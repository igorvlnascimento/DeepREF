import json
import argparse
import pandas as pd

from collections import Counter

def csv2id(dataset_name):
    train_df = pd.read_csv(f"benchmark/{dataset_name}/original/{dataset_name}_train_original.csv", sep="\t")
    val_df = pd.read_csv(f"benchmark/{dataset_name}/original/{dataset_name}_val_original.csv", sep="\t")
    test_df = pd.read_csv(f"benchmark/{dataset_name}/original/{dataset_name}_test_original.csv", sep="\t")
    
    upos_list = list(train_df['pos_tags']) + list(val_df['pos_tags']) + list(test_df['pos_tags'])
    upos_counter_dict = dict(Counter([item for sublist in [upos.split() for upos in upos_list] for item in sublist]))
    deps_list = list(train_df['dependencies_labels']) + list(val_df['dependencies_labels']) + list(test_df['dependencies_labels'])
    deps_counter_dict = dict(Counter([item for sublist in [deps.split() for deps in deps_list] for item in sublist]))
    
    upos2id = {k:i for i, (k, v) in enumerate(upos_counter_dict.items())}
    deps2id = {k:i for i, (k, v) in enumerate(deps_counter_dict.items())}
    return upos2id, deps2id

def save2json(dataset_name, upos_dict, deps_dict):
    with open(f'benchmark/{dataset_name}/{dataset_name}_upos2id.json', 'w') as json_file:
        json_file.write(json.dumps(upos_dict))
    
    with open(f'benchmark/{dataset_name}/{dataset_name}_deps2id.json', 'w') as json_file:
        json_file.write(json.dumps(deps_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='none', choices=['semeval2010', 'semeval2018', 'semeval20181-1', 'semeval20181-2', 'ddi'],
        help='Dataset. If not none, the following args can be ignored')
    
    args = parser.parse_args()
    pos2id, deps2id = csv2id(args.dataset)
    save2json(args.dataset, pos2id, deps2id)