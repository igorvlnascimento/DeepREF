import os
import json
from ast import literal_eval

import subprocess
import stanza
import spacy

import pandas as pd

from opennre import config

class ConverterDataset():
    
    def __init__(self, dataset_name, nlp_tool="stanza", nlp_model="general", entity_name="ENTITY"):
        
        self.dataset_name = dataset_name
        self.entity_name = entity_name
        
        with open(config.NLP_CONFIG, 'r') as f:
            self.nlp_config = json.load(f)
            
        self.nlp_tool = nlp_tool
            
        nlp_info = self.nlp_config[nlp_tool][nlp_model]
        
        print(nlp_info)
        
        if nlp_tool == 'stanza':
            if nlp_model == 'general':
                stanza.download('en')
                self.nlp = stanza.Pipeline(lang=nlp_info["language_model"], processors="tokenize,ner,depparse,pos,lemma", tokenize_no_ssplit=True)
            else:
                stanza.download('en', package=nlp_info["package"], processors={'ner': nlp_info["ner_model"]})
                self.nlp = stanza.Pipeline(nlp_info["language_model"], package=nlp_info["package"], processors={"ner": nlp_info["ner_model"]}, tokenize_no_ssplit=True)
        elif nlp_tool == 'spacy':
                subprocess.call(["python", "-m", "spacy", "download", nlp_info["package"]])
                self.nlp = spacy.load(nlp_info["package"])
                
        with open(config.RELATIONS_TYPE, 'r') as f:
            relations_type = json.load(f)
                
        os.makedirs(os.path.join('benchmark', self.dataset_name), exist_ok=True)
        
        self.write_relations_json(self.dataset_name, relations_type[self.dataset_name])
                
        
    def write_relations_json(self, dataset_name, relation_dict):
        json_file = open('benchmark/{}/{}_rel2id.json'.format(dataset_name, dataset_name), 'w')
        json.dump(relation_dict, json_file)

    def tokenize(self, tag_sentence):
        if self.nlp_tool == "spacy":
            doc = self.nlp(tag_sentence)
            tokenized = [token.text for token in doc]
            upos = [token.pos_ for token in doc]
            deps = [token.dep_.lower() for token in doc]
            ner = ["O"] * len(tokenized)
            for ent in doc.ents:
                for i in range(ent.start, ent.end):
                    ner[i] = ent.label_
        elif self.nlp_tool == "stanza":
            #tag_sentence_tokenized = tag_sentence.split()
            doc = self.nlp(tag_sentence)
            tokenized = [token.text for sent in doc.sentences for token in sent.words]
            upos = [token.upos for sent in doc.sentences for token in sent.words]
            deps = [token.deprel for sent in doc.sentences for token in sent.words]
            ner = [token.ner for sent in doc.sentences for token in sent.tokens]
        assert len(tokenized) == len(upos) == len(deps) == len(ner) 
        return tokenized, upos, deps, ner

    # remove any additional whitespace within a line
    def remove_whitespace(self, line):
        return str(" ".join(line.split()).strip())

    # We will replace e1 by DRUG, e2 by OTHERDRUG, common between e1 and e2 as EITHERDRUG and other drugs as 
    # UNRELATEDDRUG (TODO: edit this)
    def tag_sentence(self, sentence, e1_data, e2_data, other_entities):
        position_dict = self.create_positions_dict(e1_data, e2_data, other_entities)
        sorted_positions = self.sort_position_keys(position_dict)
        new_sentence = ''
        # TODO (geeticka): check for cases when previous ending position and next starting position 
        # are equal or next to each other. Add a space in that case
        # we are inserting a starter and ender tag around the drugs
        for i in range(len(sorted_positions)):
            curr_pos = sorted_positions[i]
            curr_start_pos, curr_end_pos = self.parse_position(curr_pos)
            if i == 0:
                new_sentence += sentence[:curr_start_pos] + ' ' + position_dict[curr_pos]['start'] + ' ' + \
                        sentence[curr_start_pos: curr_end_pos+1] + ' ' + position_dict[curr_pos]['end'] + ' '
            else:
                prev_pos = sorted_positions[i-1]
                _, prev_end_pos = self.parse_position(prev_pos)
                middle = sentence[prev_end_pos+1 : curr_start_pos]
                if middle == '':
                    middle = ' '
                new_sentence += middle + ' ' + position_dict[curr_pos]['start'] + ' ' + \
                        sentence[curr_start_pos: curr_end_pos+1] + ' ' + position_dict[curr_pos]['end'] + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                    new_sentence += ' ' + sentence[curr_end_pos+1:]
        new_sentence = self.remove_whitespace(new_sentence)
        
        return new_sentence

    # get the start and end of the entities 
    def get_entity_start_and_end(self, entity_start, entity_end, tokens):
        pass

    # given the entity starting and ending word index, and entity replacement dictionary, 
    # update the dictionary to inform of the replace_by string for eg ENTITY
    def get_entity_replacement_dictionary(self, e_idx, entity_replacement, replace_by):
        key = str(e_idx[0]) + ":" + str(e_idx[1])
        entity_replacement[key] = replace_by
        return entity_replacement
    
    #given string 12-30, return 12, 30 as a tuple of ints
    def parse_position(self, position):
        positions = position.split('-')
        return int(positions[0]), int(positions[1])
        #if metadata['e1']['charOffset'] and metadata['e2']['charOffset'] have something in common

    # given position dictionary, sort the positions from ascending order. Assumes no overlap. 
    # will be messed up if there is overlap
    # can also check for overlap but not right now
    def sort_position_keys(self, position_dict):
        positions = list(position_dict.keys())
        sorted_positions = sorted(positions, key=lambda x: int(x.split('-')[0]))
        return sorted_positions

    # TODO: need to edit this 
    def get_dataset_dataframe(self, directory=None, relation_extraction=True):
        '''
        If relation_extraction is True, then we don't care whether the ddi flag is true or false
        '''
        pass


    # to streamline the writing of the dataframe
    def write_dataframe(self, df, directory):
        df.to_csv(directory, sep='\t', encoding='utf-8', index=False)

    # to streamline the reading of the dataframe
    def read_dataframe(self, directory):
        df = pd.read_csv(directory, sep='\t')
        def literal_eval_metadata(row):
            metadata = row.metadata
            metadata = literal_eval(metadata)
            return metadata
        df['metadata'] = df.apply(literal_eval_metadata, axis=1)
        # metadata is a dictionary which is written into the csv format as a string
        # but in order to be treated as a dictionary it needs to be evaluated
        return df
    
    # The goal here is to make sure that the df that is written into memory is the same one that is read
    def check_equality_of_written_and_read_df(self, df, df_copy):
        bool_equality = df.equals(df_copy)
        # to double check, we want to check with every column
        bool_every_column = True
        for idx in range(len(df)):
            row1 = df.iloc[idx]
            row2 = df_copy.iloc[idx]
            if row1['original_sentence'] != row2['original_sentence'] or row1['e1'] != row2['e1'] or \
                    row1['relation_type'] != row2['relation_type'] or \
                    row1['tokenized_sentence'] != row2['tokenized_sentence'] or \
                    row1['metadata'] != row2['metadata']: 
                        bool_every_column = False
                        break
        return bool_equality, bool_every_column
    
    # suggested in
    # https://stackoverflow.com/questions/10632839/transform-list-of-tuples-into-a-flat-list-or-a-matrix
    def flatten_list_of_tuples(self, a):
        return list(sum(a, ()))

    # write the dataframe into the text format accepted by the cnn model
    def write_into_txt(self, df, directory):
        pass
    
    def write_split_dataframes(self, output_path, train_input_file, test_input_file):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        self.dataset_name = output_path.split('/')[1]
            
        #original_df_names = [self.dataset_name + '_{}_original.csv'.format(split) for split in ['train', 'val', 'test']]

        #for df_name in original_df_names:
        if not os.path.exists(os.path.join(output_path, self.dataset_name + '_train_original.csv')):
            df_train = self.get_dataset_dataframe(train_input_file)
            self.write_dataframe(df_train, os.path.join(output_path, self.dataset_name + '_train_original.csv'))
            df_train_copy = self.read_dataframe(os.path.join(output_path, self.dataset_name + '_train_original.csv'))
            self.check_equality_of_written_and_read_df(df_train, df_train_copy)
        # else:
        #     df_train = self.read_dataframe(os.path.join(output_path, self.dataset_name + '_train_original.csv'))
        
        if not os.path.exists(os.path.join(output_path, self.dataset_name + '_val_original.csv')):
            df_val = df_train.sample(frac=0.2, random_state=config.SEED)
            df_train = df_train.drop(df_val.index)
            self.write_dataframe(df_train, os.path.join(output_path, self.dataset_name + '_train_original.csv'))
            df_train_copy = self.read_dataframe(os.path.join(output_path, self.dataset_name + '_train_original.csv'))
            self.check_equality_of_written_and_read_df(df_val, df_train_copy)
            self.write_dataframe(df_val, os.path.join(output_path, self.dataset_name + '_val_original.csv'))
            df_val_copy = self.read_dataframe(os.path.join(output_path, self.dataset_name + '_val_original.csv'))
            self.check_equality_of_written_and_read_df(df_val, df_val_copy)
        # else:
        #     df_val = self.read_dataframe(os.path.join(output_path, self.dataset_name + '_val_original.csv'))
        
        if not os.path.exists(os.path.join(output_path, self.dataset_name + '_test_original.csv')):
            df_test = self.get_dataset_dataframe(test_input_file)
            self.write_dataframe(df_test, os.path.join(output_path, self.dataset_name + '_test_original.csv'))
            df_test_copy = self.read_dataframe(os.path.join(output_path, self.dataset_name + '_test_original.csv'))
            self.check_equality_of_written_and_read_df(df_test, df_test_copy)
        else:
            df_test = self.read_dataframe(os.path.join(output_path, self.dataset_name + '_test_original.csv'))
            
        for file in os.listdir(output_path):
            if file.endswith(".csv"):
                df = self.read_dataframe(os.path.join(output_path, file))
                self.write_into_txt(df, os.path.join(output_path, file[:file.rfind('.')] + '.txt'))
