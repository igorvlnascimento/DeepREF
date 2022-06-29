import os
import math
import copy
import json
import random
import pandas as pd

from opennre import config
from opennre.dataset.sentence import Sentence

class Dataset():
    def __init__(self, name:str, train_sentences:list=[], test_sentences:list=[], val_sentences:list=[], val_perc:float=0.2, preprocessing_type:str ="original"):
        self.name = name
        self.preprocessing_type = preprocessing_type
        self.train_sentences = train_sentences
        self.test_sentences = test_sentences
        self.val_sentences = val_sentences if val_sentences else self.split_val_sentences(val_perc)
        self.classes = self.get_classes()
        
    def set_seed(self):
        random.seed(config.SEED)
        
    def split_val_sentences(self, val_perc):
        self.set_seed()
        train_length = len(self.train_sentences)
        val_length = math.ceil(train_length*val_perc)
        train_sentences_copy = copy.deepcopy(self.train_sentences)
        random.shuffle(train_sentences_copy)
        return train_sentences_copy[:val_length]
    
    def get_classes(self):
        classes = set()
        for data in self.test_sentences:
            classes.add(data.relation_type)
        return classes
    
    def write_classes_json(self):
        json_file = open('benchmark/{}/{}_rel2id.json'.format(self.name, self.name), 'w')
        classes_dict = {k:v for v, k in enumerate(self.classes)}
        json.dump(classes_dict, json_file)
    
    def write_dataframe(self):
        train_data = []
        val_data = []
        test_data = []
        for sentence in self.train_sentences:
            train_data.append(sentence.get_sentence_info())
        for sentence in self.val_sentences:
            val_data.append(sentence.get_sentence_info())
        for sentence in self.test_sentences:
            test_data.append(sentence.get_sentence_info())
        columns = 'original_sentence,e1,e2,relation_type,pos_tags,dependencies_labels,ner,sk_entities'.split(',')
        train_df = pd.DataFrame(train_data,
            columns=columns)
        val_df = pd.DataFrame(val_data,
            columns=columns)
        test_df = pd.DataFrame(test_data,
            columns=columns)
        train_df.to_csv(f"benchmark/{self.name}/original/{self.name}_train_original.csv", sep='\t', encoding='utf-8', index=False)
        val_df.to_csv(f"benchmark/{self.name}/original/{self.name}_val_original.csv", sep='\t', encoding='utf-8', index=False)
        test_df.to_csv(f"benchmark/{self.name}/original/{self.name}_test_original.csv", sep='\t', encoding='utf-8', index=False)
        
    def write_text(self, preprocessing_types=[]):
        if len(preprocessing_types):
            preprocessing_types = sorted(preprocessing_types)
            preprocessing_types = "_".join(preprocessing_types)
        else:
            preprocessing_types = "original"
        train_data = []
        val_data = []
        test_data = []
        for sentence in self.train_sentences:
            sentence_dict = self.set_sentence_dict(sentence.get_sentence_info())
            train_data.append(sentence_dict)
        for sentence in self.val_sentences:
            sentence_dict = self.set_sentence_dict(sentence.get_sentence_info())
            val_data.append(sentence_dict)
        for sentence in self.test_sentences:
            sentence_dict = self.set_sentence_dict(sentence.get_sentence_info())
            test_data.append(sentence_dict)
        os.makedirs(f'benchmark/{self.name}/{preprocessing_types}/', exist_ok=True)
        with open(f'benchmark/{self.name}/{preprocessing_types}/{self.name}_train_{preprocessing_types}.txt', 'w') as f:
            for data in train_data:
                f.write(str(data)+'\n')
        with open(f'benchmark/{self.name}/{preprocessing_types}/{self.name}_val_{preprocessing_types}.txt', 'w') as f:
            for data in val_data:
                f.write(str(data)+'\n')
        with open(f'benchmark/{self.name}/{preprocessing_types}/{self.name}_test_{preprocessing_types}.txt', 'w') as f:
            for data in test_data:
                f.write(str(data)+'\n')
                
    def set_sentence_dict(self, sentence_info):
        sentence_dict = {}
        sentence_dict["token"] = sentence_info[0].split()
        sentence_dict["h"] = {'name': sentence_info[1]['name'], 'pos': sentence_info[1]['position']}
        sentence_dict["t"] = {'name': sentence_info[2]['name'], 'pos': sentence_info[2]['position']}
        sentence_dict["relation"] = sentence_info[3]
        sentence_dict["pos_tags"] = sentence_info[4].split()
        sentence_dict["deps"] = sentence_info[5].split()
        sentence_dict["ner"] = sentence_info[6].split()
        sentence_dict["sk"] = sentence_info[7]
        return sentence_dict
        
    def load_dataset_csv(self):
         train_df = pd.read_csv(f'benchmark/{self.name}/original/{self.name}_train_original.csv', sep='\t')
         val_df = pd.read_csv(f'benchmark/{self.name}/original/{self.name}_val_original.csv', sep='\t')
         test_df = pd.read_csv(f'benchmark/{self.name}/original/{self.name}_test_original.csv', sep='\t')
         train_sentences = self.csv_to_sentences(train_df)
         val_sentences = self.csv_to_sentences(val_df)
         test_sentences = self.csv_to_sentences(test_df)
         self.train_sentences = train_sentences
         self.test_sentences = test_sentences
         self.val_sentences = val_sentences
         return self
    
    def csv_to_sentences(self, dataframe: pd.DataFrame):
        sentences = []
        for i in range(len(dataframe.index)):
            sentence = Sentence('', '')
            sentence.load_sentence(*dataframe.iloc[[i]].values.flatten().tolist())
            sentences.append(sentence)
        return sentences
