import os
import argparse
import stanza

from opennre.dataset.preprocess import Preprocess
from opennre.dataset.converters.converter import ConverterDataset

indir = 'original/'
outdir1 = 'entity_blinding/'
outdir2 = 'punct_stop_digit/'
outdir3 = 'punct_digit/'
outdir4 = 'ner_blinding/'

class PreprocessDataset():
    def __init__(self, dataset_name, preprocessing_types, nlp):
        self.dataset_name = dataset_name
        self.output_path = 'benchmark/' + dataset_name + '/'
        self.preprocessing_types = preprocessing_types
        self.nlp = nlp
        
        self.preprocess_dataset()

    def out(self, path): return os.path.join(self.output_path, path)

    def output_file_length(self, filename):
        return len(open(self.out(filename)).readlines())

    def preprocess_dataset(self):
        
        preprocessing_types = {
            "entity_blinding": False, 
            "digit": False, 
            "punct": False, 
            "stopword": False, 
            "brackets": False,
            "ner_blinding": False, 
            "reverse_sentences": False, 
            "semantic_features": False
        }
        
        if "all" in self.preprocessing_types:
            preprocessing_types = {
                "entity_blinding": False, 
                "digit": True, 
                "punct": True, 
                "stopword": True, 
                "ner_blinding": True, 
                "reverse_sentences": True, 
                "syntatic_features": False,
                "semantic_features": True
            }
        else:
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
            if "rs" in self.preprocessing_types:
                preprocessing_types["reverse_sentences"] = True
            if "sf" in self.preprocessing_types:
                preprocessing_types["semantic_features"] = True
        
        preprocess = Preprocess(self.dataset_name, preprocessing_types, self.nlp)

        original_dataframe_names = [self.dataset_name + '_train', self.dataset_name + '_val']
        # self.makedir(outdir1)
        # self.makedir(outdir2)
        # self.makedir(outdir3)
        # self.makedir(outdir4)

        for original_df_name in original_dataframe_names:
            #preprocess.preprocessing_types["entity_blinding"] = True
            #type1 = preprocess.preprocess(converter.read_dataframe, self.out("original/" + original_df_name + '_original.csv'))
            #preprocess.preprocessing_types["digit"] = True
            #preprocess.preprocessing_types["punct"] = True
            #preprocess.preprocessing_types["stopword"] = True
            #type2 = preprocess.preprocess(converter.read_dataframe, self.out("original/" + original_df_name + '_original.csv'))
            #preprocess.preprocessing_types["stopword"] = False
            #type3 = preprocess.preprocess(converter.read_dataframe, self.out("original/" + original_df_name + '_original.csv'))
            #preprocess.preprocessing_types["ner_blinding"] = True
            type4 = preprocess.preprocess(self.out("original/" + original_df_name + '_original.csv'))
            # converter.write_dataframe(type1, out(outdir1 + original_df_name + '_entity_blinding.csv'))
            # converter.write_dataframe(type2, out(outdir2 + original_df_name + '_punct_stop_digit.csv'))
            # converter.write_dataframe(type3, out(outdir3 + original_df_name + '_punct_digit.csv'))
            # converter.write_dataframe(type4, out(outdir4 + original_df_name + '_ner_blinding.csv'))
            #converter.write_into_txt(type1, self.out(outdir1 + original_df_name + '_entity_blinding.txt'))
            #converter.write_into_txt(type2, self.out(outdir2 + original_df_name + '_punct_stop_digit.txt'))
            #converter.write_into_txt(type3, self.out(outdir3 + original_df_name + '_punct_digit.txt'))
            preprocess.write_into_txt(type4, self.out(outdir4 + original_df_name + '_ner_blinding.txt'))

        # for original_df_name in original_dataframe_names:
        #     type1 = converter.read_dataframe(out(outdir1 + original_df_name + '_entity_blinding.csv'))
        #     type2 = converter.read_dataframe(out(outdir2 + original_df_name + '_punct_stop_digit.csv'))
        #     type3 = converter.read_dataframe(out(outdir3 + original_df_name + '_punct_digit.csv'))
        #     type4 = converter.read_dataframe(out(outdir4 + original_df_name + '_ner_blinding.csv'))
        #     converter.write_into_txt(type1, out(outdir1 + original_df_name + '_entity_blinding.txt'))
        #     converter.write_into_txt(type2, out(outdir2 + original_df_name + '_punct_stop_digit.txt'))
        #     converter.write_into_txt(type3, out(outdir3 + original_df_name + '_punct_digit.txt'))
        #     converter.write_into_txt(type4, out(outdir4 + original_df_name + '_ner_blinding.txt'))

        # print(self.output_file_length(self.out, indir + converter.dataset_name + '_train_original.txt'))
        # print(self.output_file_length(self.out, outdir1 + converter.dataset_name + '_train_entity_blinding.txt'))
        # print(self.output_file_length(self.out, outdir2 + converter.dataset_name + '_train_punct_stop_digit.txt'))
        # print(self.output_file_length(self.out, outdir3 + converter.dataset_name + '_train_punct_digit.txt'))
        # print(self.output_file_length(self.out, outdir4 + converter.dataset_name + '_train_ner_blinding.txt'))

        # print(self.output_file_length(self.out, indir + converter.dataset_name + '_val_original.txt'))
        # print(self.output_file_length(self.out, outdir1 + converter.dataset_name + '_val_entity_blinding.txt'))
        # print(self.output_file_length(self.out, outdir2 + converter.dataset_name + '_val_punct_stop_digit.txt'))
        # print(self.output_file_length(self.out, outdir3 + converter.dataset_name + '_val_punct_digit.txt'))
        # print(self.output_file_length(self.out, outdir4 + converter.dataset_name + '_val_ner_blinding.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
        help='Dataset name')
    parser.add_argument('-p', '--preprocessing_types', type=str, default="all",
        help='Preprocessing types')
    
    args = parser.parse_args()
    
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors="tokenize,ner,mwt,pos", tokenize_no_ssplit=True)
    
    preprocess_dataset = PreprocessDataset(args.dataset_name, args.preprocessing_types, nlp)
    #preprocess_dataset(args.dataset_name, args.output_path, args.preprocessing_types)


