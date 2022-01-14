import os
import argparse
import subprocess
from opennre import constants

from opennre.dataset.preprocess import Preprocess

class PreprocessDataset():
    def __init__(self, dataset_name, preprocessing_types):
        self.dataset_name = dataset_name
        self.preprocessing_types = None if preprocessing_types is None else sorted(preprocessing_types)
        self.preprocessing_types_str = 'original' if preprocessing_types is None else "_".join(self.preprocessing_types)
        self.output_path = os.path.join('benchmark', dataset_name, self.preprocessing_types_str)

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
        
        if self.preprocessing_types is not None:
            if "nb" in self.preprocessing_types:
                preprocessing_types["ner_blinding"] = True
            elif "eb" in self.preprocessing_types:
                preprocessing_types["entity_blinding"] = True
                
            if "d" in self.preprocessing_types:
                preprocessing_types["digit"] = True
            if "p" in self.preprocessing_types:
                preprocessing_types["punct"] = True
            if "sw" in self.preprocessing_types:
                preprocessing_types["stopword"] = True
            if "b" in self.preprocessing_types:
                preprocessing_types["brackets"] = True
    
        preprocess = Preprocess(self.dataset_name, preprocessing_types)

        original_dataframe_names = [self.dataset_name + '_train', self.dataset_name + '_val', self.dataset_name + '_test']
        self.makedir()

        for original_df_name in original_dataframe_names:
            if not os.path.exists(os.path.join('benchmark', self.dataset_name, 'original', original_df_name + '_original.txt')) or \
                not os.path.exists(os.path.join('benchmark', self.dataset_name, 'original', original_df_name + '_original.csv')):
                cmd = ['bash', 'benchmark/download_{}.sh'.format(self.dataset_name)]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                _, _ = proc.communicate()
            if not os.path.exists(self.out(original_df_name + '_{}.txt'.format(self.preprocessing_types_str))):
                print("preprocessing_types_str:",self.preprocessing_types_str)
                original_ds = preprocess.preprocess(os.path.join('benchmark', self.dataset_name, 'original', original_df_name + '_original.csv'))
                preprocess.write_into_txt(original_ds, self.out(original_df_name + '_{}.txt'.format(self.preprocessing_types_str)))
            
        for original_df_name in original_dataframe_names:
            print(self.output_file_length(os.path.join('benchmark', self.dataset_name, 'original', '{}_original.txt'.format(original_df_name))))
            print(self.output_file_length(self.out('{}_{}.txt'.format(original_df_name, self.preprocessing_types_str))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, required=True, choices=constants.DATASETS,
        help='Dataset name')
    parser.add_argument('-p', '--preprocessing_types', nargs="+", choices=constants.PREPROCESSING_TYPES,
        help='Preprocessing types')
    
    args = parser.parse_args()
        
    if args.preprocessing_types is not None:
        preprocess_dataset = PreprocessDataset(args.dataset_name, args.preprocessing_types)
        preprocess_dataset.preprocess_dataset()
    else:
        for comb in constants.preprocessing_choices:
            print("comb:",comb)
            preprocess_dataset = PreprocessDataset(args.dataset_name, comb)
            preprocess_dataset.preprocess_dataset()


