import os
import argparse
import stanza
import subprocess

from opennre.dataset.preprocess import Preprocess

class PreprocessDataset():
    def __init__(self, dataset_name, preprocessing_types, nlp):
        self.dataset_name = dataset_name
        self.preprocessing_types = None if preprocessing_types is None else sorted(preprocessing_types)
        self.preprocessing_types_str = 'original' if preprocessing_types is None else "_".join(self.preprocessing_types)
        self.output_path = os.path.join('benchmark', dataset_name, self.preprocessing_types_str)
        self.nlp = nlp

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
            "ner_blinding": False, 
            "reverse_sentences": False, 
            "semantic_features": False,
            "syntatic_features": False
        }
        
        if self.preprocessing_types is not None:
            if "nb" in self.preprocessing_types:
                preprocessing_types["ner_blinding"] = True
            elif "eb" in self.preprocessing_types:
                preprocessing_types["entity_bliding"] = True
                
            if "d" in self.preprocessing_types:
                preprocessing_types["digit"] = True
            if "p" in self.preprocessing_types:
                preprocessing_types["punct"] = True
            if "sw" in self.preprocessing_types:
                preprocessing_types["stopword"] = True
            if "b" in self.preprocessing_types:
                preprocessing_types["brackets"] = True
            if "rs" in self.preprocessing_types:
                preprocessing_types["reverse_sentences"] = True
            if "syf" in self.preprocessing_types:
                preprocessing_types["syntatic_features"] = True
            if "sf" in self.preprocessing_types:
                preprocessing_types["semantic_features"] = True
    
            preprocess = Preprocess(self.dataset_name, preprocessing_types, self.nlp)

        original_dataframe_names = [self.dataset_name + '_train', self.dataset_name + '_val', self.dataset_name + '_test']
        self.makedir()

        for original_df_name in original_dataframe_names:
            if not os.path.exists(os.path.join('benchmark', self.dataset_name, 'original', original_df_name + '_original.txt')) or \
                not os.path.exists(os.path.join('benchmark', self.dataset_name, 'original', original_df_name + '_original.csv')):
                cmd = ['bash', 'benchmark/download_{}.sh'.format(self.dataset_name)]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                _, _ = proc.communicate()
            original_ds = preprocess.preprocess(os.path.join('benchmark', self.dataset_name, 'original', original_df_name + '_original.csv'))
            preprocess.write_into_txt(original_ds, self.out(original_df_name + '_{}.txt'.format(self.preprocessing_types_str)))
            
        for original_df_name in original_dataframe_names:
            print(self.output_file_length(os.path.join('benchmark', self.dataset_name, 'original', '{}_original.txt'.format(original_df_name))))
            print(self.output_file_length(self.out('{}_{}.txt'.format(original_df_name, self.preprocessing_types_str))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
        help='Dataset name')
    parser.add_argument('-p', '--preprocessing_types', nargs="+", default="all",
        help='Preprocessing types')
    
    args = parser.parse_args()
    
    if args.dataset_name == "semeval2010":
        stanza.download('en')
        nlp = stanza.Pipeline(lang='en', processors="tokenize,ner,pos", tokenize_no_ssplit=True)
    else:
        stanza.download('en', package='craft', processors={'ner': 'bionlp13cg'})
        nlp = stanza.Pipeline('en', package="craft", processors={"ner": "bionlp13cg"}, tokenize_no_ssplit=True)
    
    preprocess_dataset = PreprocessDataset(args.dataset_name, args.preprocessing_types, nlp)
    preprocess_dataset.preprocess_dataset()


