'''
Author: Geeticka Chauhan
Performs pre-processing on a csv file independent of the dataset (once converters have been applied). 
Refer to notebooks/Data-Preprocessing for more details. The methods are specifically used in the non
_original notebooks for all datasets.
'''

import os
import pandas as pd
from ast import literal_eval
from tqdm.auto import tqdm

import nltk
from nltk.corpus import stopwords

class Preprocess():
    def __init__(self, dataset_name, preprocessing_types, entity_name="ENTITY") -> None:
        self.dataset_name = dataset_name
        self.preprocessing_types = preprocessing_types
        # important global variables for identifying the location of entities
        self.entity1 = 'E'
        self.entity2 = 'EOTHER'
        self.entity_either = 'EEITHER'
        
        self.entity_name = entity_name
        
        self.output_path = os.path.join("benchmark", dataset_name)
        
        nltk.download('stopwords')

    '''
    The methods below are for the preprocessing type 1 
    '''
    # separate the indexes of entity 1 and entity 2 by what is intersecting 
    # and what is not
    def get_common_and_separate_entities(self, e1_indexes, e2_indexes):
        e1_indexes = set(e1_indexes[0])
        e2_indexes = set(e2_indexes[0])
        common_indexes = e1_indexes.intersection(e2_indexes)
        only_e1_indexes = list(e1_indexes.difference(common_indexes))
        only_e2_indexes = list(e2_indexes.difference(common_indexes))
        
        return only_e1_indexes, only_e2_indexes, list(common_indexes)


    # given an entity replacement dictionary like {'0:0': 'entity1'} 
    # provide more information related to the location of the entity
    def entity_replacement_dict_with_entity_location(self, entity_replacement_dict, replacement_type,
                                                    only_e1_indexes, only_e2_indexes, common_indexes):
        def update_dict_with_indexes(new_entity_replacement_dict, only_indexes, start, end):
            for i in only_indexes:
                key = str(i[0]) + ':' + str(i[-1])
                new_entity_replacement_dict[key]['start'] = start
                new_entity_replacement_dict[key]['end'] = end
            return new_entity_replacement_dict
            
        new_entity_replacement_dict = {} 
        # below is just for initialization purposes, when start and end is none, means we are not 
        # inserting anything before or after those words in the sentence
        for key in entity_replacement_dict.keys():
                new_entity_replacement_dict[key] = {'replace_by': entity_replacement_dict[key], 
                                            'start': None, 'end': None}
        new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, only_e1_indexes,
                                                            self.entity1 + 'START', self.entity1 + 'END')
        new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, only_e2_indexes,
                                                            self.entity2 + 'START', self.entity2 + 'END')
        new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, common_indexes,
                                                            self.entity_either + 'START', self.entity_either + 'END')
        return new_entity_replacement_dict
    
    # given an entity replacement dictionary like {'0:0': 'entity1'} 
    # provide more information related to the location of the entity
    def entity_replacement_dict_with_entity_location_embed(self, entity_replacement_dict, embed,
                                                    only_e1_indexes, only_e2_indexes, common_indexes):
        def update_dict_with_indexes(new_entity_replacement_dict, only_indexes):
            for i in only_indexes:
                key = str(i[0]) + ':' + str(i[-1])
            return new_entity_replacement_dict
            
        new_entity_replacement_dict = {} 
        # below is just for initialization purposes, when start and end is none, means we are not 
        # inserting anything before or after those words in the sentence
        for key in entity_replacement_dict.keys():
            idx = int(key.split(':')[-1])
            new_entity_replacement_dict[key] = {'replace_by': embed[idx], 
                                        'start': None, 'end': None}
        new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, only_e1_indexes)
        new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, only_e2_indexes)
        new_entity_replacement_dict = update_dict_with_indexes(new_entity_replacement_dict, common_indexes)
        return new_entity_replacement_dict

    ###
    ### Helper functions
    ###
    #given string 12:30, return 12, 30 as a tuple of ints
    def parse_position(self, position):
        positions = position.split(':')
        return int(positions[0]), int(positions[1])

    def sort_position_keys(self, entity_replacement_dict):
        positions = list(entity_replacement_dict.keys())
        sorted_positions = sorted(positions, key=lambda x: int(x.split(':')[0]))
        return sorted_positions

    # remove any additional whitespace within a line
    def remove_whitespace(self, line):
        return str(" ".join(line.split()).strip())

    def list_to_string(self, sentence):
        return " ".join(sentence)

    # adapted from tag_sentence method in converter_ddi
    # note that white spaces are added in the new sentence on purpose
    def replace_with_concept(self, row, replacement_type):
        sentence = row.tokenized_sentence.split(" ")
        upos = row.upos_sentence.split(" ")
        deps = row.deps_sentence.split(" ")
        ner = row.ner_sentence.split(" ")
        e1_indexes = row.metadata['e1']['word_index']
        e2_indexes = row.metadata['e2']['word_index'] # assuming that within the same entity indexes, no overlap
        new_sentence = ''
        new_upos = ''
        new_deps = ''
        new_ner = ''
        only_e1_indexes, only_e2_indexes, common_indexes = \
        self.get_common_and_separate_entities(e1_indexes, e2_indexes)
        
        entity_replacement_dict = row.metadata['entity_replacement'] # assuming no overlaps in replacement
        
        new_entity_replacement_dict = self.entity_replacement_dict_with_entity_location(entity_replacement_dict, replacement_type,
                                                                                only_e1_indexes, only_e2_indexes, 
                                                                                common_indexes)
        repl_dict = new_entity_replacement_dict # just using proxy because names are long
        
        new_entity_replacement_dict_upos = self.entity_replacement_dict_with_entity_location_embed(entity_replacement_dict, upos,
                                                                                only_e1_indexes, only_e2_indexes, 
                                                                                common_indexes)
        repl_dict_upos = new_entity_replacement_dict_upos # just using proxy because names are long
        
        new_entity_replacement_dict_deps = self.entity_replacement_dict_with_entity_location_embed(entity_replacement_dict, deps,
                                                                                only_e1_indexes, only_e2_indexes, 
                                                                                common_indexes)
        repl_dict_deps = new_entity_replacement_dict_deps # just using proxy because names are long
        
        new_entity_replacement_dict_ner = self.entity_replacement_dict_with_entity_location_embed(entity_replacement_dict, ner,
                                                                                only_e1_indexes, only_e2_indexes, 
                                                                                common_indexes)
        repl_dict_ner = new_entity_replacement_dict_ner # just using proxy because names are long
        sorted_positions = self.sort_position_keys(new_entity_replacement_dict)
        for i in range(len(sorted_positions)):
            curr_pos = sorted_positions[i]
            curr_start_pos, curr_end_pos = self.parse_position(curr_pos)
            start_replace = '' if repl_dict[curr_pos]['start'] is None else repl_dict[curr_pos]['start'].upper()
            end_replace = '' if repl_dict[curr_pos]['end'] is None else repl_dict[curr_pos]['end'].upper()
            between_replace = repl_dict[curr_pos]['replace_by'][replacement_type].upper() # between the entity replacement
            
            start_replace_other = ''
            end_replace_other = ''
            
            between_replace_upos = repl_dict_upos[curr_pos]['replace_by'] # between the entity replacement
            between_replace_deps = repl_dict_deps[curr_pos]['replace_by'] # between the entity replacement
            between_replace_ner = repl_dict_ner[curr_pos]['replace_by'] # between the entity replacement
            
            
            if i == 0:
                new_sentence += self.list_to_string(sentence[:curr_start_pos]) + ' ' + start_replace + ' ' + \
                between_replace + ' ' + end_replace + ' '
                
                new_upos += self.list_to_string(upos[:curr_start_pos]) + ' ' + start_replace_other + ' ' + \
                between_replace_upos + ' ' + end_replace_other + ' '
                
                new_deps += self.list_to_string(deps[:curr_start_pos]) + ' ' + start_replace_other + ' ' + \
                between_replace_deps + ' ' + end_replace_other + ' '
                
                new_ner += self.list_to_string(ner[:curr_start_pos]) + ' ' + start_replace_other + ' ' + \
                between_replace_ner + ' ' + end_replace_other + ' '
            else:
                prev_pos = sorted_positions[i-1]
                _, prev_end_pos = self.parse_position(prev_pos)
                middle = self.list_to_string(sentence[prev_end_pos+1 : curr_start_pos]) # refers to middle between prev segment and the 
                middle_upos = self.list_to_string(upos[prev_end_pos+1 : curr_start_pos]) # refers to middle between prev segment and the 
                middle_deps = self.list_to_string(deps[prev_end_pos+1 : curr_start_pos]) # refers to middle between prev segment and the 
                middle_ner = self.list_to_string(ner[prev_end_pos+1 : curr_start_pos]) # refers to middle between prev segment and the 
                # current segment
                if middle == '':
                    middle = ' '
                new_sentence += middle + ' ' + start_replace + ' ' + between_replace + ' ' + end_replace + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                    new_sentence += ' ' + self.list_to_string(sentence[curr_end_pos+1:])
                    
                if middle_upos == '':
                    middle_upos = ' '
                new_upos += middle_upos + ' ' + start_replace_other + ' ' + between_replace_upos + ' ' + end_replace_other + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(upos) - 1:
                    new_upos += ' ' + self.list_to_string(upos[curr_end_pos+1:])
                    
                if middle_deps == '':
                    middle_deps = ' '
                new_deps += middle_deps + ' ' + start_replace_other + ' ' + between_replace_deps + ' ' + end_replace_other + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(deps) - 1:
                    new_deps += ' ' + self.list_to_string(deps[curr_end_pos+1:])
                    
                if middle_ner == '':
                    middle_ner = ' '
                new_ner += middle_ner + ' ' + start_replace_other + ' ' + between_replace_ner + ' ' + end_replace_other + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(ner) - 1:
                    new_ner += ' ' + self.list_to_string(ner[curr_end_pos+1:])
        new_sentence = self.remove_whitespace(new_sentence)
        new_upos = self.remove_whitespace(new_upos)
        new_deps = self.remove_whitespace(new_deps)
        new_ner = self.remove_whitespace(new_ner)
        return pd.Series([new_sentence, new_upos, new_deps, new_ner])


    '''
    Preprocessing Type 2: Removal of stop words, punctuations and the replacement of digits
    '''
    # gives a dictionary signifying the location of the different entities in the sentence
    def get_entity_location_dict(self, only_e1_indexes, only_e2_indexes, common_indexes):
        entity_location_dict = {}
        def update_dict_with_indexes(entity_location_dict, only_indexes, start, end):
            for i in only_indexes:
                key = str(i[0]) + ':' + str(i[-1])
                entity_location_dict[key] = {'start': start, 'end': end}
            return entity_location_dict
        entity_location_dict = update_dict_with_indexes(entity_location_dict, only_e1_indexes, 
                                                        self.entity1 + 'START', self.entity1 + 'END')
        entity_location_dict = update_dict_with_indexes(entity_location_dict, only_e2_indexes, 
                                                        self.entity2 + 'START', self.entity2 + 'END')
        entity_location_dict = update_dict_with_indexes(entity_location_dict, common_indexes, 
                                                        self.entity_either + 'START', self.entity_either + 'END')
        return entity_location_dict

    # given the index information of the entities, return the sentence with 
    # tags ESTART EEND etc to signify the location of the entities
    def get_new_sentence_with_entity_replacement(self, sentence, e1_indexes, e2_indexes):
        new_sentence = ''
        only_e1_indexes, only_e2_indexes, common_indexes = \
            self.get_common_and_separate_entities(e1_indexes, e2_indexes)  
        entity_loc_dict = self.get_entity_location_dict(only_e1_indexes, only_e2_indexes, common_indexes)
        sorted_positions = self.sort_position_keys(entity_loc_dict)
        for i in range(len(sorted_positions)):
            curr_pos = sorted_positions[i]
            curr_start_pos, curr_end_pos = self.parse_position(curr_pos)
            start_replace = entity_loc_dict[curr_pos]['start']
            end_replace = entity_loc_dict[curr_pos]['end']
            if i == 0:
                new_sentence += self.list_to_string(sentence[:curr_start_pos]) + ' ' + start_replace + ' ' + \
                self.list_to_string(sentence[curr_start_pos : curr_end_pos + 1]) + ' ' + end_replace + ' '
            else:
                prev_pos = sorted_positions[i-1]
                _, prev_end_pos = self.parse_position(prev_pos)
                middle = self.list_to_string(sentence[prev_end_pos+1 : curr_start_pos])
                if middle == '':
                    middle = ' '
                new_sentence += middle + ' ' + start_replace + ' ' + \
                        self.list_to_string(sentence[curr_start_pos: curr_end_pos+1]) + ' ' + end_replace + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                    new_sentence += ' ' + self.list_to_string(sentence[curr_end_pos+1:])
        new_sentence = self.remove_whitespace(new_sentence)
        # TODO write some code to do the replacement
        return new_sentence

    # preprocessing 2: remove the stop words and punctuation from the data
    # and replace all digits
    # TODO: might be nice to give an option to specify whether to remove the stop words or not 
    # this is a low priority part though
    def generate_new_sentence(self, sentence, upos, deps, ner, index_dict):
        new_sentence = []
        new_upos = []
        new_deps = []
        new_ner = []
        if isinstance(sentence, str):
            sentence = sentence.split(" ")
        idx = 0
        entity_start_or_end = 0
        for i in range(len(sentence)):
            word = sentence[i]
            if word.endswith('END') or word.endswith('START'):
                new_sentence.append(word)
                entity_start_or_end += 1
                continue
            if not index_dict[i]['keep']:
                idx += 1
                continue # don't append when it is a stop word or punctuation
            if index_dict[i]['replace_with'] is not None:
                words_length = len(index_dict[i]['replace_with'].split('_'))
                new_sentence.extend(index_dict[i]['replace_with'].split('_'))
                new_upos.extend([upos[idx]]*words_length)
                new_deps.extend([deps[idx]]*words_length)
                new_ner.extend([ner[idx]]*words_length)
                idx += 1
                continue
            new_sentence.append(word)
            new_upos.append(upos[idx])
            new_deps.append(deps[idx])
            new_ner.append(ner[idx])
            idx += 1
        assert len(new_sentence) == len(new_upos) + entity_start_or_end and \
                len(new_sentence) == len(new_deps) + entity_start_or_end and \
                len(new_sentence) == len(new_ner) + entity_start_or_end
        return self.list_to_string(new_sentence), self.list_to_string(new_upos), self.list_to_string(new_deps), self.list_to_string(new_ner)
    
    def generate_tagged_sentence(self, row):
        sentence = row.tokenized_sentence.split(" ")
        e1_indexes = row.metadata['e1']['word_index'][0]
        e2_indexes = row.metadata['e2']['word_index'][0]
        if not 'ESTART' in sentence and not 'EEND' in sentence:
            sentence = self.get_new_sentence_with_entity_replacement(sentence, e1_indexes, e2_indexes)
        
        return sentence
    
    def replace_digit_punctuation_stop_word_brackets(self, row):
        if "tagged_sentence" in row:
            sentence = row.tagged_sentence.split(" ")
        else:
            sentence = row.tokenized_sentence.split(" ")
        upos = row.upos_sentence.split(" ")
        deps = row.deps_sentence.split(" ")
        ner = row.ner_sentence.split(" ")
        e1_indexes = row.metadata['e1']['word_index']
        e2_indexes = row.metadata['e2']['word_index']
        if not 'ESTART' in sentence and not 'EEND' in sentence:
            sentence = self.get_new_sentence_with_entity_replacement(sentence, e1_indexes, e2_indexes)
        
        if isinstance(sentence,str):
            sentence = sentence.split(" ")
            
        # detection of stop words, punctuations and digits
        index_to_keep_dict = {} # index: {keep that token or not, replace_with}
        stop_words = set(stopwords.words('english'))
        stop_words.remove('o')
        brackets = False
        idx = 0
        for i, word in enumerate(sentence):
            is_entity = False
            word_index = i
            for tup in e1_indexes:
                if word_index >= tup[0] and word_index <= tup[1]:
                    is_entity = True
                    index_to_keep_dict[i] = {'keep': True, 'replace_with': None}
            for tup in e2_indexes:
                if word_index >= tup[0] and word_index <= tup[1]:
                    is_entity = True
                    index_to_keep_dict[i] = {'keep': True, 'replace_with': None}
            if is_entity:
                #idx += 1
                continue
            stop_word = word in stop_words and self.preprocessing_types["stopword"]
            punct = not word.endswith('END') and upos[idx] == 'PUNCT' and self.preprocessing_types["punct"]
            num = not word.endswith('END') and upos[idx] == "NUM" and self.preprocessing_types["digit"]
            if not brackets and self.preprocessing_types["brackets"]:
                brackets = "(" == word or "[" == word
            if stop_word:
                index_to_keep_dict[word_index] = {'keep': False, 'replace_with': None}
            elif brackets:
                index_to_keep_dict[word_index] = {'keep': False, 'replace_with': None}
                brackets = not (")" == word) and not ("]" == word)
            elif punct:
                index_to_keep_dict[word_index] = {'keep': False, 'replace_with': None}
            elif num:
                index_to_keep_dict[word_index] = {'keep': True, 'replace_with': 'NUMBER'}
            else:
                index_to_keep_dict[word_index] = {'keep': True, 'replace_with': None}
            
            if not word.endswith('START') and not word.endswith('END') and idx < len(upos) - 1:
                idx += 1
        # generation of the new sentence based on the above findings
        new_sentence, new_upos, new_deps, new_ner = self.generate_new_sentence(sentence, upos, deps, ner, index_to_keep_dict)
        return pd.Series([new_sentence, new_upos, new_deps, new_ner])

    '''
    Preprocessing Type 3 part 1: NER
    '''

    # a method to check for overlap between the ner_dict that is created
    def check_for_overlap(self, ner_dict):
        def expand_key(string): # a string that looks like '2:2' to [2]
            start = int(string.split(':')[0])
            end = int(string.split(':')[1])
            return list(range(start, end+1))
        expanded_keys = [expand_key(key) for key in ner_dict.keys()]
        for i1, item in enumerate(expanded_keys):
            for i2 in range(i1 + 1, len(expanded_keys)):
                if set(item).intersection(expanded_keys[i2]):
                    return True # overlap is true
            for i2 in range(0, i1):
                if set(item).intersection(expanded_keys[i2]):
                    return True
        return False


    ###
    ### Helper functions for the NER replacement
    ###
    def overlap_index(self, index1, index2):
        def expand(index):
            start = int(index[0])
            end = int(index[1])
            return list(range(start, end+1))
        expand_index1 = expand(index1)
        expand_index2 = expand(index2)
        if set(expand_index1).intersection(set(expand_index2)):
            return True
        else: return False
        
    # for indexes that look like (1,1) and (2,2) check if the left is fully included in the right
    def fully_included(self, index1, index2):
        if int(index1[0]) >= int(index2[0]) and int(index1[1]) <= int(index2[1]): return True
        else: return False

    def beginning_overlap(self, index1, index2): # this is tricky when (1,1) and (2,2) are there
        if int(index1[0]) < int(index2[0]) and int(index1[1]) <= int(index2[1]): return True
        else: return False

    def end_overlap(self, index1, index2): # this is tricky
        if int(index1[0]) >= int(index2[0]) and int(index1[1]) > int(index2[1]): return True
        else: return False
        
    def beginning_and_end_overlap(self, index1, index2):
        if int(index1[0]) < int(index2[0]) and int(index1[1]) > int(index2[1]): return True
        else:
            return False
    #else there is no overlap

    # taken from https://stackoverflow.com/questions/46548902/converting-elements-of-list-of-nested-lists-from-string-to-integer-in-python
    def list_to_int(self, lists):
        return [int(el) if not isinstance(el,list) else self.convert_to_int(el) for el in lists]

    def correct_entity_indexes_with_ner(self, ner_dict, e_index):
        for i in range(len(e_index)): # we are reading tuples here
            for key in ner_dict.keys():
                indexes = e_index[i]
                index2 = indexes
                index1 = self.parse_position(key) # checking if ner is fully included etc
                if not self.overlap_index(index1, index2): # don't do below if there is no overlap
                    continue
                if self.beginning_overlap(index1, index2):
                    e_index[i] = (index1[0], e_index[i][1])
                elif self.end_overlap(index1, index2):
                    e_index[i] = (e_index[i][0], index1[1])
                elif self.beginning_and_end_overlap(index1, index2):
                    e_index[i] = (index1[0], index1[1]) # else you don't change or do anything
        return e_index

    # given all of these dictionaries, return the ner replacement dictionary
    def get_ner_replacement_dictionary(self, only_e1_index, only_e2_index, common_indexes, ner_dict):
        def update_dict_with_entity(e_index, ner_repl_dict, entity_name):
            for indexes in e_index:
                key1 = str(indexes[0]) + ':' + str(indexes[0])
                key1start = str(indexes[0]) + ':' + str(indexes[0]) + ':' + entity_name + 'START'
                ner_repl_dict[key1] = {'replace_by': ner_dict[key1], 'insert': None}
                ner_repl_dict[key1start] = {'replace_by': None, 'insert': entity_name + 'START'}
                
                key2 = str(int(indexes[-1]) - 1) + ':' + str(int(indexes[-1]) - 1)
                key2end = str(int(indexes[-1]) - 1) + ':' + str(int(indexes[-1]) - 1) + ':' + entity_name + 'END'
                
                ner_repl_dict[key2] = {'replace_by': ner_dict[key2], 'insert': None}
                ner_repl_dict[key2end] = {'replace_by': None, 'insert': entity_name + 'END'}
            return ner_repl_dict
        # we are going to do something different: only spans for NER will be counted, but
        # for the ENTITYSTART and ENTITYEND, we will keep the span as what token to insert before
        ner_repl_dict = {}
        ner_repl_dict = update_dict_with_entity(only_e1_index, ner_repl_dict, self.entity1)
        ner_repl_dict = update_dict_with_entity(only_e2_index, ner_repl_dict, self.entity2)
        ner_repl_dict = update_dict_with_entity(common_indexes, ner_repl_dict, self.entity_either)
        return ner_repl_dict

    # this function is different from the sort_position_keys because
    # we care about sorting not just by the beginning token, but also by the length that the span contains
    def ner_sort_position_keys(self, ner_repl_dict): # this can potentially replace sort_position_keys
        # but only if the application of this function does not change the preprocessed CSVs generated
        def len_key(key):
            pos = self.parse_position(key)
            return pos[1] - pos[0] + 1
        def start_or_end(key): 
            # handle the case where the ending tag of the entity is in the same place as the
            #starting tag of another entity - this happens when two entities are next to each other
            if len(key.split(':')) <= 2: # means that this is a named entity
                return 3
            start_or_end = key.split(':')[2]
            if start_or_end.endswith('END'): # ending spans should get priority
                return 1
            elif start_or_end.endswith('START'):
                return 2
        positions = list(ner_repl_dict.keys())
        sorted_positions = sorted(positions, key=lambda x: (self.parse_position(x)[0], len_key(x), start_or_end(x)))
        return sorted_positions

    # given a splitted sentence - make sure that the sentence is in list form
    def get_ner_dict(self, ner_sentence):        
        ner_tokenized = ner_sentence.split(" ")
        ner_dict = {} # first test for overlaps within ner
        for i, ner in enumerate(ner_tokenized):
            key = str(i) + ':' + str(i)
            ner_dict[key] = ner
        return ner_dict

    def convert_indexes_to_int(self, e_idx):
        new_e_idx = []
        for indexes in e_idx:
            t = (int(indexes[0]), int(indexes[1]))
            new_e_idx.append(t)
        return new_e_idx

    def replace_ner(self, row, check_ner_overlap=False): # similar to concept_replace, with some caveats
        sentence = row.tokenized_sentence.split()
        e1_indexes = row.metadata['e1']['word_index'][0]
        e2_indexes = row.metadata['e2']['word_index'][0]
        e1_indexes = self.convert_indexes_to_int(e1_indexes)
        e2_indexes = self.convert_indexes_to_int(e2_indexes)
        only_e1_indexes, only_e2_indexes, common_indexes = \
        self.get_common_and_separate_entities(e1_indexes, e2_indexes)
        ner_dict = self.get_ner_dict(row.ner_sentence)
        if check_ner_overlap and self.check_for_overlap(ner_dict):
            print("There is overlap", ner_dict) # only need to check this once
        #Below code works only if there isn't overlap within ner_dict, so make sure that there isn't overlap
        
        # overlaps between ner label and e1 and e2 indexes are a problem
        # And they can be of two types
            # Type 1: NER overlaps with e1 or e2 in the beginning or end
            # Here we want to keep the NER link the same but extend e1 or e2 index to the beginning or end of the
            # NER 

            #Type 2: NER is inside of the entity completely: At this point it should be simply ok to mention at what 
            # token to insert ENTITYstart and ENTITYend
            # Type 1 is a problem, but Type 2 is easy to handle while the new sentence is being created
            
        only_e1_indexes = self.correct_entity_indexes_with_ner(ner_dict, only_e1_indexes)
        only_e2_indexes = self.correct_entity_indexes_with_ner(ner_dict, only_e2_indexes)
        common_indexes = self.correct_entity_indexes_with_ner(ner_dict, common_indexes)

        # below needs to be done in case there was again a shift that might have caused both e1 and e2 to have
        # the same spans
        only_e1_indexes, only_e2_indexes, common_indexes2 = \
                self.get_common_and_separate_entities(only_e1_indexes, only_e2_indexes)
        common_indexes.extend(common_indexes2)

        ner_repl_dict = self.get_ner_replacement_dictionary(only_e1_indexes, only_e2_indexes, common_indexes,
                                                    ner_dict)
        sorted_positions = self.ner_sort_position_keys(ner_repl_dict)
        new_sentence = ''
        for i in range(len(sorted_positions)):
            curr_pos = sorted_positions[i]
            curr_start_pos, curr_end_pos = self.parse_position(curr_pos)
            curr_dict = ner_repl_dict[curr_pos]
            start_insert = '' if curr_dict['insert'] is None else curr_dict['insert'].upper()
            between_replace = '' if curr_dict['replace_by'] is None else curr_dict['replace_by']
            if i == 0:
                new_sentence += self.list_to_string(sentence[:curr_start_pos]) + ' ' + start_insert + ' ' + \
                between_replace + ' '
            else:
                prev_pos = sorted_positions[i-1]
                _, prev_end_pos = self.parse_position(prev_pos)
                if ner_repl_dict[prev_pos]['insert'] is None: # means middle will be starting from prev_pos + 1
                    middle = self.list_to_string(sentence[prev_end_pos+1 : curr_start_pos])
                else: # means middle needs to start from the prev_pos
                    middle = self.list_to_string(sentence[prev_end_pos: curr_start_pos])
                if middle == '':
                    middle = ' '
                new_sentence += middle + ' ' + start_insert + ' ' + between_replace + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                    position = curr_end_pos + 1 if curr_dict['insert'] is None else curr_end_pos
                    new_sentence += ' ' + self.list_to_string(sentence[position:])
                    
        new_sentence = self.remove_whitespace(new_sentence)
        return new_sentence

    '''
    Below methods do entity detection from the tagged sentences, i.e. a sentence that contains 
    ESTART, EEND etc, use that to detect the locations of the respective entities and remove the tags
    from the sentence to return something clean
    '''
    # below is taken directly from the ddi converter and 
    # removes the first occurence of the start and end, and tells of their location
    def get_entity_start_and_end(self, entity_start, entity_end, tokens):
        e_start = tokens.index(entity_start)
        e_end = tokens.index(entity_end) - 2 # 2 tags will be eliminated

        between_tags = 0
        for index in range(e_start + 1, e_end + 2): 
            # we want to check between the start and end for occurence of other tags
            if tokens[index].endswith('START') or tokens[index].endswith('END'):
                between_tags += 1
        e_end -= between_tags

        # only eliminate the first occurence of the entity_start and entity_end
        new_tokens = []
        entity_start_seen = 0
        entity_end_seen = 0
        for i, x in enumerate(tokens):
            if x == entity_start:
                entity_start_seen += 1
            if x == entity_end:
                entity_end_seen += 1
            if x == entity_start and entity_start_seen == 1:
                continue
            if x == entity_end and entity_end_seen == 1:
                continue
            new_tokens.append(x)
        return (e_start, e_end), new_tokens


    # based upon the method in converter for DDI, this will do removal of the entity tags and keep 
    # track of where they are located in the sentence
    def get_entity_positions_and_replacement_sentence(self, tokens):
        e1_idx = []
        e2_idx = []
        
        tokens_for_indexing = tokens
        for token in tokens:
            if token.endswith('START') and len(token) > len('START'):
                ending_token = token[:-5] + 'END'
                e_idx, tokens_for_indexing = \
                    self.get_entity_start_and_end(token, ending_token, tokens_for_indexing)
                if token == self.entity1 + 'START' or token == self.entity_either + 'START':
                    e1_idx.append(e_idx)
                if token == self.entity2 + 'START' or token == self.entity_either + 'START':
                    e2_idx.append(e_idx)
        return e1_idx, e2_idx, tokens_for_indexing


    '''
    Returns the dataframe after doing the preprocessing 
    '''

    # update the metadata and the sentence with the preprocessed version
    def update_metadata_sentence(self, row):
        tagged_sentence = row.tagged_sentence
        e1_idx, e2_idx, tokens_for_indexing = \
            self.get_entity_positions_and_replacement_sentence(tagged_sentence.split())
        new_sentence = self.list_to_string(tokens_for_indexing)
        metadata = row.metadata
        metadata['e1']['word_index'] = e1_idx
        metadata['e2']['word_index'] = e2_idx
        metadata['e1']['word'] = " ".join(tokens_for_indexing[e1_idx[0][0]: e1_idx[0][1]+1])
        metadata['e2']['word'] = " ".join(tokens_for_indexing[e2_idx[0][0]: e2_idx[0][1]+1])
        metadata.pop('entity_replacement', None) # remove the entity replacement dictionary from metadata
        row.tokenized_sentence = new_sentence
        row.metadata = metadata
        return row
    
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
    
    # give this preprocessing function a method to read the dataframe, and the location of the original 
    # dataframe to read so that it can do the preprocessing
    # whether to do type 1 vs type 2 of the preprocessing
    # 1: replace with all concepts in the sentence, 2: replace the stop words, punctuations and digits
    # 3: replace only punctuations and digits
    def preprocess(self, df_directory):
        tqdm.pandas()
        df = self.read_dataframe(df_directory)
        
        none = True
        for p in self.preprocessing_types:
            if self.preprocessing_types[p]:
                none = False
                break
            
        if none:
            df['tagged_sentence'] = df.progress_apply(self.generate_tagged_sentence, axis=1)
            
            print("Updating metadata sentence:")
            df = df.progress_apply(self.update_metadata_sentence, axis=1)
        else:
            if self.preprocessing_types["ner_blinding"]:
                print("NER blinding preprocessing:")
                df[['tagged_sentence', 'upos_sentence', 'deps_sentence', 'ner_sentence']] = df.progress_apply(self.replace_with_concept, replacement_type='ner', axis=1)
                
                print("Updating metadata sentence:")
                df = df.progress_apply(self.update_metadata_sentence, axis=1)
            elif self.preprocessing_types["entity_blinding"]:
                print("Entity blinding preprocessing:")
                df[['tagged_sentence', 'upos_sentence', 'deps_sentence', 'ner_sentence']] = df.progress_apply(self.replace_with_concept, replacement_type='entity', axis=1) # along the column axis
                
                print("Updating metadata sentence:")
                df = df.progress_apply(self.update_metadata_sentence, axis=1)
                
            if self.preprocessing_types["digit"] or \
                self.preprocessing_types["punct"] or \
                self.preprocessing_types["stopword"] or \
                self.preprocessing_types["brackets"]:
                    print("Digit, punctuaction, brackets, stopword or wordnet preprocessing:")
                    df[['tagged_sentence', 'upos_sentence', 'deps_sentence', 'ner_sentence']] = \
                        df.progress_apply(self.replace_digit_punctuation_stop_word_brackets, axis=1)
                
                    print("Updating metadata sentence:")
                    df = df.progress_apply(self.update_metadata_sentence, axis=1)
            
        df = df.rename({'tokenized_sentence': 'preprocessed_sentence'}, axis=1)
        df = df.drop(['tagged_sentence'], axis=1)
        return df
    
    def flatten_list_of_tuples(self, a):
        return list(sum(a, ()))    
    
    def write_into_txt(self, df, directory):
        print("Unique relations: \t", df['relation_type'].unique())
        null_row = df[df["relation_type"].isnull()]
        if null_row.empty:
            idx_null_row = None
        else:
            idx_null_row = null_row.index.values[0]
        with open(directory, 'w') as outfile:
            for i in tqdm(range(0, len(df))):
                dict = {}
                head = {}
                tail = {}
                if idx_null_row is not None and i == idx_null_row:
                    continue
                row = df.iloc[i]
                metadata = row.metadata
                # TODO: need to change below in order to contain a sorted list of the positions
                e1 = self.flatten_list_of_tuples(metadata['e1']['word_index'])
                e2 = self.flatten_list_of_tuples(metadata['e2']['word_index'])
                e1 = sorted(e1)
                e2 = sorted(e2)
                head["name"] = metadata['e1']['word']
                head["pos"] = [e1[0], e1[1]+1]
                tail["name"] = metadata['e2']['word']
                tail["pos"] = [e2[0], e2[1]+1]
                try:
                    tokenized_sentence = row.tokenized_sentence
                except AttributeError:
                    tokenized_sentence = row.preprocessed_sentence
                if type(tokenized_sentence) is not str:
                    continue
                tokenized_sentence = tokenized_sentence.split(" ")
                dict["token"] = tokenized_sentence
                dict["h"] = head
                dict["t"] = tail
                dict["pos"] = row.upos_sentence.split(" ")
                dict["deps"] = row.deps_sentence.split(" ")
                dict["ner"] = row.ner_sentence.split(" ")
                dict["sk"] = row.sk
                dict["relation"] = row.relation_type
                outfile.write(str(dict)+"\n")
            outfile.close()
