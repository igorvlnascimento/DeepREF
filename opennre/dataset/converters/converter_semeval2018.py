import os
import glob
import argparse
import stanza

import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from tqdm import tqdm

from opennre.dataset.converters.converter import ConverterDataset

relation_dict = {0: 'usage', 1: 'result', 2: 'model-feature', 3: 'part_whole', 4: 'topic', 5: 'compare'}
rev_relation_dict = {val: key for key, val in relation_dict.items()}

class ConverterSemEval2018(ConverterDataset):
    def __init__(self, nlp) -> None:
        super().__init__(dataset_name="semeval2018", entity_name="ENTITY", nlp=nlp)
        
        os.makedirs(os.path.join('benchmark', self.dataset_name), exist_ok=True)
        
        self.write_relations_json(self.dataset_name, rev_relation_dict)
        
    def tokenize(self, sentence, model="spacy"):
        if model == "spacy":
            doc = self.nlp(sentence)
            tokenized = []
            for token in doc:
                tokenized.append(token.text)
            return tokenized
        elif model == "stanza":
            doc = self.nlp(sentence)
            tokenized = [token.text for sent in doc.sentences for token in sent.tokens]
            return tokenized

    # given sentence dom in DDI corpus, get all the information related to the entities 
    # present in the dom
    def get_entity_dict(self, sentence):
        if '<abstract>' in sentence:
            abstract_start = sentence.find('<abstract>')
            abstract_end = sentence.find('</abstract>')
            sentence = sentence[abstract_start:abstract_end]
            
            sentence = sentence[sentence.find('>')+1:]
            
            
        if '<title>' in sentence:
            title_start = sentence.find('<title>')
            title_end = sentence.find('</title>')
            sentence = sentence[title_start:title_end]
            
            sentence = sentence[sentence.find('>')+1:]
        entity_dict = {}
        while sentence.find('<entity') != -1:
            entity_start = sentence.find("<entity")
            entity_end = sentence.find("</entity>")
            id_end = sentence.find('>')
            entity_id = sentence[entity_start+12:id_end-1]
            
            entity_name = sentence[id_end+1:entity_end]
            
            sentence = sentence[:entity_start] + sentence[id_end+1:]
            charOffset = ["{}-{}".format(len(sentence[:entity_start]), len(sentence[:entity_start])+len(entity_name)-1)]
            entity_end = sentence.find("</entity>")
            sentence = sentence[:entity_end] + sentence[entity_end+9:]
            print("sentence:",sentence)
            if '<entity' in sentence[:entity_end]:
                break
            entity_dict[entity_id] = {'id': entity_id, 'word': entity_name, 'charOffset':charOffset}
        return entity_dict, sentence


    def get_pairs(self, entity_dict, entity_pairs):
        pairs = []
        entity_list = [key for key in entity_dict]
        for entity in entity_list:
            if entity in entity_pairs:
                pairs.append((entity_pairs[entity]['relation'], entity, entity_pairs[entity]['e2']))
        return pairs
    
    # given the metadata, get the individual positions in the sentence and know what to replace them by
    def create_positions_dict(self, e1, e2, other_entities):
        position_dict = {}
        for pos in e1['charOffset']:
            if pos not in position_dict:
                position_dict[pos] = {'start': self.entity_name+'START', 'end': self.entity_name+'END'}
        for pos in e2['charOffset']:
            if pos not in position_dict:
                position_dict[pos] = {'start': self.entity_name+'OTHERSTART', 'end': self.entity_name+'OTHEREND'}
        for other_ent in other_entities:
            for pos in other_ent['charOffset']:
                if pos not in position_dict:
                    position_dict[pos] = {'start': self.entity_name+'UNRELATEDSTART', 'end': self.entity_name+'UNRELATEDEND'}
        return position_dict

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

    # return the entities in the sentence except those in the pair
    def get_other_entities(self, entity_dict, e1, e2):
        blacklisted_set = [e1, e2]
        return [value for key, value in entity_dict.items() if key not in blacklisted_set]

    # get the start and end of the entities 
    def get_entity_start_and_end(self, entity_start, entity_end, tokens):
        e_start = tokens.index(entity_start)
        e_end = tokens.index(entity_end) - 2 # because 2 tags will be eliminated
        # only eliminate the entity_start and entity_end once because DRUGUNRELATEDSTART will get
        # eliminated many times
        new_tokens = []
        entity_start_seen = 0
        entity_end_seen = 0
        for x in tokens:
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
    # from the tokenized sentence which contains the drug tags, extract the word positions
    # and replacement dictionary for blinding purposes
    def get_entity_positions_and_replacement_dictionary(self, tokens):
        entity_replacement = {}
        e1_idx = []
        e2_idx = []

        tokens_for_indexing = tokens
        for token in tokens:
            if token.startswith('ENTITY') and token.endswith('START'):
                ending_token = token[:-5] + 'END'
                e_idx, tokens_for_indexing = self.get_entity_start_and_end(token, ending_token, tokens_for_indexing)

                replace_by = token[:-5]
                entity_replacement = self.get_entity_replacement_dictionary(e_idx, entity_replacement, replace_by)
                if token == 'ENTITYSTART' or token == 'ENTITYEITHERSTART':
                    e1_idx.append(e_idx)
                if token == 'ENTITYOTHERSTART' or token == 'ENTITYEITHERSTART':
                    e2_idx.append(e_idx)
        return e1_idx, e2_idx, entity_replacement, tokens_for_indexing

    # TODO: need to edit this 
    def get_dataset_dataframe(self, directory=None, relation_extraction=True):
        '''
        If relation_extraction is True, then we don't care whether the ddi flag is true or false
        '''
        
        punk_param = PunktParameters()
        abbreviations = ['e.g', 'viz', 'al']
        punk_param.abbrev_types = set(abbreviations)
        tokenizer = PunktSentenceTokenizer(punk_param)
        
        relation_files_to_read = glob.glob(directory + '*.txt')
        entity_pairs = {}
        for file_name in relation_files_to_read:
            lines = open(file_name).readlines()
            for line in lines:
                type = line[:line.find('(')]
                if 'REVERSE' in line:
                    e2_id = line[line.find('(')+1:line.find(',')]
                    e1_id = line[line.find(',')+1:line.find(',REVERSE)')]
                    entity_pairs[e1_id] = {'relation':type, 'e2':e2_id}
                else:
                    e1_id = line[line.find('(')+1:line.find(',')]
                    e2_id = line[line.find(',')+1:line.find(')')]
                    entity_pairs[e1_id] = {'relation':type, 'e2':e2_id}
        
        data = []
        total_files_to_read = glob.glob(directory + '*.xml')
        print('total_files_to_read:' , len(total_files_to_read) , ' from dir: ' , directory)
        for file in tqdm(total_files_to_read):
            lines = open(file).readlines()
            for line in lines:
                print("line:",line)
                if '</entity>' in line:
                    sentences = tokenizer.tokenize(line)
                    #print("sentences:",sentences)
                    for sentence in sentences:
                        entity_dict, sentence = self.get_entity_dict(sentence)
                        
                        pairs = self.get_pairs(entity_dict,entity_pairs)
                        
                        print("entity_dict:",entity_dict)
                        print("sentence:",sentence)
                        print("pairs:",pairs)
                        
                        # pegar lista de pares de entidade e relação e verificar se há pares nas sentenças

                        for pair in pairs:
                            #pair_id = pair.getAttribute('id')
                            #ddi_flag = pair.getAttribute('ddi')
                            e1_id = pair[1]
                            e2_id = pair[2]
                            
                            other_entities = self.get_other_entities(entity_dict, e1_id, e2_id)
                            
                            try:
                                e1_data = entity_dict[e1_id]
                                e2_data = entity_dict[e2_id]
                            except KeyError:
                                # TODO: tratar os coreference resolution
                                continue
                            
                            tagged_sentence = self.tag_sentence(sentence, e1_data, e2_data, other_entities)
                            tokens = self.tokenize(tagged_sentence, model="stanza")

                            e1_idx, e2_idx, entity_replacement, tokens_for_indexing = \
                                    self.get_entity_positions_and_replacement_dictionary(tokens)
                            # TODO (geeticka): for unifying purposes, can remove the e1_id and e2_id
                            metadata = {'e1': {'word': e1_data['word'], 'word_index': e1_idx, 'id': e1_id},
                                        'e2': {'word': e2_data['word'], 'word_index': e2_idx, 'id': e2_id},
                                        'entity_replacement': entity_replacement}
                            tokenized_sentence = " ".join(tokens_for_indexing)

                            relation_type = pair[0].lower()
                            if not not relation_type: # not of empty string is True, but we don't want to append
                                data.append([str(sentence), str(e1_data['word']), str(e2_data['word']),
                                    str(relation_type), metadata, str(tokenized_sentence)])

        df = pd.DataFrame(data,
                columns='original_sentence,e1,e2,relation_type,metadata,tokenized_sentence'.split(','))
        return df
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input_file', default='benchmark/raw_semeval2018/Train/', 
        help='Input path of training examples')
    parser.add_argument('--test_input_file', default='benchmark/raw_semeval2018/Test/', 
        help='Input path of training examples')
    parser.add_argument('--output_path', default='benchmark/semeval2018/original', 
        help='Input path of training examples')

    args = parser.parse_args()
    
    stanza.download('en', package='craft', processors={'ner': 'bionlp13cg'})
    nlp = stanza.Pipeline('en', package="craft", processors={"ner": "bionlp13cg"}, tokenize_no_ssplit=True)
    
    converter = ConverterSemEval2018(nlp)
    
    converter.write_split_dataframes(args.output_path, args.train_input_file, args.test_input_file)