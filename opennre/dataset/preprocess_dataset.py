import os
import json
import argparse
import subprocess
from opennre import config

from opennre.dataset.preprocess import Preprocess

class PreprocessDataset():
    def __init__(self, dataset_name, preprocessing_type):
        self.dataset_name = dataset_name
        self.preprocessing_type = []
        if len(preprocessing_type):
            self.preprocessing_type = sorted(preprocessing_type)
        self.preprocessing_type_str = "_".join(self.preprocessing_type)
        self.output_path = os.path.join('benchmark', dataset_name, self.preprocessing_type_str)

    def out(self, path): return os.path.join(self.output_path, path)
    
    def makedir(self):
        if not os.path.exists(os.path.join(self.output_path)):
            os.makedirs(os.path.join(self.output_path))

    def output_file_length(self, filename_path):
        return len(open(filename_path).readlines())

    def preprocess_dataset(self):
        
        preprocessing_types = {
            "entity_blinding": False, 
            "digit": False, 
            "punct": False, 
            "stopword": False, 
            "brackets": False,
            "ner_blinding": False
        }
        
        if self.preprocessing_type is not None:
            if "nb" in self.preprocessing_type:
                preprocessing_types["ner_blinding"] = True
            elif "eb" in self.preprocessing_type:
                preprocessing_types["entity_blinding"] = True
                
            if "d" in self.preprocessing_type:
                preprocessing_types["digit"] = True
            if "p" in self.preprocessing_type:
                preprocessing_types["punct"] = True
            if "sw" in self.preprocessing_type:
                preprocessing_types["stopword"] = True
            if "b" in self.preprocessing_type:
                preprocessing_types["brackets"] = True
    
        preprocess = Preprocess(self.dataset_name, preprocessing_types)

        original_dataframe_names = [self.dataset_name + '_train', self.dataset_name + '_val', self.dataset_name + '_test']
        self.makedir()

        for original_df_name in original_dataframe_names:
            ds_path = os.path.join('benchmark', self.dataset_name, 'original', original_df_name)
            if not os.path.exists(os.path.join(ds_path + '_original.txt')) or \
                not os.path.exists(os.path.join(ds_path + '_original.csv')):
                print("preprocessing_type_str:",self.preprocessing_type_str)
                subprocess.call(['bash', 'benchmark/download_{}.sh'.format(self.dataset_name)])
            if not os.path.exists(self.out(original_df_name + '_{}.txt'.format(self.preprocessing_type_str))):
                print("Preprocessing...")
                original_ds = preprocess.preprocess(os.path.join(ds_path + '_original.csv'))
                preprocess.write_into_txt(original_ds, self.out(original_df_name + '_{}.txt'.format(self.preprocessing_type_str)))
            
        for original_df_name in original_dataframe_names:
            print(self.output_file_length(os.path.join('benchmark', self.dataset_name, 'original', '{}_original.txt'.format(original_df_name))))
            print(self.output_file_length(self.out('{}_{}.txt'.format(original_df_name, self.preprocessing_type_str))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=config.DATASETS,
        help='Dataset name')
    parser.add_argument('-p', '--preprocessing', type=list, required=True, choices=config.PREPROCESSING_COMBINATION, nargs='+',
        help='Preprocessing types')
    
    args = parser.parse_args()
    
    with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
        best_hparams = json.load(f)
        
    preprocess_dataset = PreprocessDataset(args.dataset, args.preprocessing)
    preprocess_dataset.preprocess_dataset()


