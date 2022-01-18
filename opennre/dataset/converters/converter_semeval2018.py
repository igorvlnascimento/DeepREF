import os
import glob
import argparse
import subprocess
import stanza
import spacy

from opennre import config

import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from tqdm import tqdm

from opennre.dataset.converters.converter import ConverterDataset

class ConverterSemEval2018(ConverterDataset):
    def __init__(self, nlp_tool, nlp_tool_type) -> None:
        super().__init__(dataset_name="semeval2018", entity_name="ENTITY", nlp_tool=nlp_tool, nlp_tool_type=nlp_tool_type)

    # given sentence dom in DDI corpus, get all the information related to the entities 
    # present in the dom
    def get_entity_dict(self, sentence):
        ## error on sentence in test dataset from semeval 2018 subtask 1.1 where there is 'topicality' word
        sentence = sentence.replace("<SectionTitle></SectionTitle>", "")
        if '<abstract>' in sentence:
            abstract_start = sentence.find('<abstract>')
            abstract_end = sentence.find('</abstract>')
            if abstract_end == -1:
                abstract_end = sentence.find('</ abstract >')
            sentence = sentence[abstract_start:abstract_end]
            
            sentence = sentence[sentence.find('>')+1:]
            
        if '</abstract>' in sentence:
            abstract_end = sentence.find('</abstract>')
            sentence = sentence[:abstract_end]
            
        if '<title>' in sentence:
            title_length = len('<title>')
            title_start = sentence.find('<title>')
            title_end = sentence.find('</title>')
            if title_end == -1:
                title_end = sentence.find('</ title >')
            sentence = sentence[title_start+title_length:title_end]
            
        if '</title>' in sentence:
            title_end = sentence.find('</title>')
            sentence = sentence[:title_end]
            
        entity_dict = {}
        while sentence.find('<entity id=') != -1:
            entity_start = sentence.find("<entity")
            entity_end = sentence.find("</entity>")
            id_end = sentence.find('">')
            entity_id = sentence[entity_start+12:id_end]
            
            entity_name = sentence[id_end+2:entity_end]
            if '<entity' in entity_name:
                entity_start_in = entity_name.find("<entity")
                entity_end_in = entity_name.find("</entity>")
                id_end_in = entity_name.find('">')
                entity_id_in = entity_name[entity_start_in+12:id_end_in-1]
                
                entity_name_in = entity_name[id_end_in+2:] + entity_name[:entity_end_in]
                entity_name = entity_name[:entity_start_in] + entity_name[id_end_in+2:]
                charOffset_in = ["{}-{}".format(len(sentence[:entity_start])+entity_start_in+1, len(sentence[:entity_start])+entity_start_in+1+len(entity_name)-1)]
                entity_dict[entity_id_in] = {'id': entity_id_in, 'word': entity_name_in, 'charOffset':charOffset_in}
            
            sentence = sentence[:entity_start] + sentence[id_end+2:]
            charOffset = ["{}-{}".format(len(sentence[:entity_start]), len(sentence[:entity_start])+len(entity_name)-1)]
            entity_end = sentence.find("</entity>")
            sentence = sentence[:entity_end] + sentence[entity_end+9:]
            
            entity_dict[entity_id] = {'id': entity_id, 'word': entity_name, 'charOffset':charOffset}
            
        if '</entity>' in sentence:
            entity_end = sentence.find('</entity>')
            sentence = sentence[:entity_end]
            
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
    def get_entity_start_and_end(self, entity_start, entity_end, tokens, upos, deps, ner):
        e_start = tokens.index(entity_start)
        e_end = tokens.index(entity_end) - 2 # because 2 tags will be eliminated
        # only eliminate the entity_start and entity_end once because DRUGUNRELATEDSTART will get
        # eliminated many times
        new_tokens = []
        new_upos = []
        new_deps = []
        new_ner = []
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
            new_upos.append(upos[i])
            new_deps.append(deps[i])
            new_ner.append(ner[i])
        assert len(new_tokens) == len(new_upos) == len(new_deps) == len(new_ner)
        return (e_start, e_end), new_tokens, new_upos, new_deps, new_ner
    # from the tokenized sentence which contains the drug tags, extract the word positions
    # and replacement dictionary for blinding purposes
    def get_entity_positions_and_replacement_dictionary(self, tokens, upos, deps, ner):
        entity_replacement = {}
        e1_idx = []
        e2_idx = []
        
        # wrong_tokenization = ['n-gram', 'N-gram', 'type-unification']
        # right_tokenization = [['n', '-', 'gram'], ['N', '-', 'gram'], ['type', '-', 'unification']]

        tokens_for_indexing = tokens
        upos_for_indexing = upos
        deps_for_indexing = deps
        ner_for_indexing = ner
        for token in tokens:
            if token.startswith('ENTITY') and token.endswith('START'):
                ending_token = token[:-5] + 'END'
                e_idx, tokens_for_indexing, upos_for_indexing, deps_for_indexing, ner_for_indexing = \
                    self.get_entity_start_and_end(token, ending_token, tokens_for_indexing, upos_for_indexing, deps_for_indexing, ner_for_indexing)

                ner = None
                for i in range(e_idx[-1], e_idx[0], -1):
                    if ner_for_indexing[i] != 'O':
                        ner = ner_for_indexing[i]
                        
                replace_by = {'entity': token[:-5], 'ner': (ner if ner is not None else 'O')}
                entity_replacement = self.get_entity_replacement_dictionary(e_idx, entity_replacement, replace_by)
                if token == 'ENTITYSTART' or token == 'ENTITYEITHERSTART':
                    e1_idx.append(e_idx)
                if token == 'ENTITYOTHERSTART' or token == 'ENTITYEITHERSTART':
                    e2_idx.append(e_idx)
        return e1_idx, e2_idx, entity_replacement, tokens_for_indexing, upos_for_indexing, deps_for_indexing, ner_for_indexing
    
    
    def reverse_sentence(self, tokens):
        idx_start = []
        idx_end = []
        entities = []
        for i, t in enumerate(tokens):
            if t.startswith('ENTITY') and t.endswith('START'):
                idx_start.append(i)
            if t.startswith('ENTITY') and t.endswith('END'):
                idx_end.append(i)
                entities.append(tokens[idx_start[-1]:idx_end[-1]+1])
        
        for i, idx in enumerate(list(zip(idx_start, idx_end))[::-1]):
            tokens = tokens[:idx[0]] + ['ENTITY' + str(i+1)] + tokens[idx[1]+1:]
            
        
        punct = tokens[-1]
        
        tokens = tokens[:-1]
        tokens = tokens[::-1] + [punct]
        
        ent_idx = []
        for i, t in enumerate(tokens):
            if t.startswith('ENTITY'):
                ent_idx.append(i)
                
        ent_idx = ent_idx[::-1]
                
        for i, idx in enumerate(ent_idx):
            tokens = tokens[:idx] + entities[i] + tokens[idx+1:]
        
        return tokens
    
    def remove_tags(self, tokens):
        for t in range(len(tokens)-1, -1, -1):
            if tokens[t].startswith('ENTITY'):
                tokens.remove(tokens[t])
                
        return tokens

    # TODO: need to edit this 
    def get_dataset_dataframe(self, directory=None, relation_extraction=True):
        '''
        If relation_extraction is True, then we don't care whether the ddi flag is true or false
        '''
        
        punk_param = PunktParameters()
        abbreviations = ['e.g', 'viz', 'al']
        punk_param.abbrev_types = set(abbreviations)
        tokenizer = PunktSentenceTokenizer(punk_param)
        
        relation_files_to_read = glob.glob(directory + '**/*.txt', recursive=True)
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
        total_files_to_read = glob.glob(directory + '**/*.xml', recursive=True)
        print('total_files_to_read:' , len(total_files_to_read) , ' from dir: ' , directory)
        for file in total_files_to_read:
            lines = open(file).readlines()
            for line in tqdm(lines):
                line = line.strip().strip('\"').strip()
                if '</entity>' in line:
                    sentences = tokenizer.tokenize(line)
                                
                    for sentence in sentences:
                        sentence = sentence.strip().strip('\"').strip()
                        sentence = sentence.strip()
                        #print('sentence1:' , sentence)
                        entity_dict, sentence = self.get_entity_dict(sentence)
                        #print('sentence2:' , sentence)
                        
                        pairs = self.get_pairs(entity_dict,entity_pairs)
                        
                        #print("entity_dict:",entity_dict)
                        # print("sentence:",sentence)
                        # print("pairs:",pairs)
                        
                        # pegar lista de pares de entidade e relação e verificar se há pares nas sentenças
                        key_error = 0
                        for pair in pairs:
                            e1_id = pair[1]
                            e2_id = pair[2]
                            
                            other_entities = self.get_other_entities(entity_dict, e1_id, e2_id)
                            
                            if e1_id in entity_dict and e2_id in entity_dict:
                                e1_data = entity_dict[e1_id]
                                e2_data = entity_dict[e2_id]
                            else:
                                # print("key_error:",key_error)
                                # print("sentence:",sentence)
                                # print("entity_dict:",entity_dict)
                                # print("e1_id:",e1_id)
                                # print("e2_id:",e2_id)
                                continue
                            
                            tagged_sentence = self.tag_sentence(sentence, e1_data, e2_data, other_entities)
                            tokens, upos, deps, ner = self.tokenize(tagged_sentence)
                            
                            # rev_tokens = self.reverse_sentence(tokens)
                            # rev_tokens_copy = rev_tokens.copy()
                               
                            # rev_sentence = " ".join(self.remove_tags(rev_tokens_copy))
                            
                            #print("tokens:",rev_tokens)
                            #if not reverse:
                            e1_idx, e2_idx, entity_replacement, tokens_for_indexing, upos_for_indexing, deps_for_indexing, ner_for_indexing = \
                                    self.get_entity_positions_and_replacement_dictionary(tokens, upos, deps, ner)
                                
                            assert len(tokens_for_indexing) == len(upos_for_indexing) \
                                and len(tokens_for_indexing) == len(deps_for_indexing) \
                                and len(tokens_for_indexing) == len(ner_for_indexing)
                            # rev_e1_idx, rev_e2_idx, rev_entity_replacement, rev_tokens_for_indexing = \
                            #         self.get_entity_positions_and_replacement_dictionary(rev_tokens)
                            # # TODO (geeticka): for unifying purposes, can remove the e1_id and e2_id
                            metadata = {'e1': {'word': e1_data['word'], 'word_index': e1_idx, 'id': e1_id},
                                        'e2': {'word': e2_data['word'], 'word_index': e2_idx, 'id': e2_id},
                                        'entity_replacement': entity_replacement}
                            
                            # TODO (geeticka): for unifying purposes, can remove the e1_id and e2_id
                            # rev_metadata = {'e1': {'word': e1_data['word'], 'word_index': rev_e1_idx, 'id': e1_id},
                            #             'e2': {'word': e2_data['word'], 'word_index': rev_e2_idx, 'id': e2_id},
                            #             'entity_replacement': rev_entity_replacement}
                            
                            tokenized_sentence = " ".join(tokens_for_indexing)
                            new_upos = " ".join(upos_for_indexing)
                            new_deps = " ".join(deps_for_indexing)
                            new_ner = " ".join(ner_for_indexing)
                            
                            # rev_tokenized_sentence = " ".join(rev_tokens_for_indexing)
                            
                            #print("rev_sentence:",rev_sentence)

                            #if reverse:
                            relation_type = pair[0].lower()
                            if not not relation_type: # not of empty string is True, but we don't want to append
                                data.append([str(sentence.lower()), str(e1_data['word']).lower(), str(e2_data['word']).lower(),
                                        str(relation_type), metadata, str(tokenized_sentence.lower()), new_upos, new_deps, new_ner])
                                #if reverse:
                                #    data.append([str(rev_sentence.lower()), str(e1_data['word']).lower(), str(e2_data['word']).lower(),
                                #            str(relation_type), rev_metadata, str(rev_tokenized_sentence.lower())])
                            # else:
                            #     relation_type = 'none'
                            #     if not not relation_type: # not of empty string is True, but we don't want to append
                            #         data.append([str(sentence.lower()), str(e1_data['word']).lower(), str(e2_data['word']).lower(),
                            #             str(relation_type), metadata, str(tokenized_sentence.lower())])
                                
                            # else:
                            #     rev_tokens = self.reverse_sentence(tokens)
                               
                            #     rev_sentence = " ".join(rev_tokens)
                               
                            #     e1_idx, e2_idx, entity_replacement, tokens_for_indexing = \
                            #             self.get_entity_positions_and_replacement_dictionary(rev_tokens)
                            #     # TODO (geeticka): for unifying purposes, can remove the e1_id and e2_id
                            #     rev_metadata = {'e1': {'word': e1_data['word'], 'word_index': e1_idx, 'id': e1_id},
                            #                 'e2': {'word': e2_data['word'], 'word_index': e2_idx, 'id': e2_id},
                            #                 'entity_replacement': entity_replacement}
                            #     tokenized_sentence = " ".join(tokens_for_indexing)

                            #     relation_type = pair[0].lower()
                            #     if not not relation_type: # not of empty string is True, but we don't want to append
                            #         data.append([str(rev_sentence.lower()), str(e1_data['word']).lower(), str(e2_data['word']).lower(),
                            #             str(relation_type), rev_metadata, str(tokenized_sentence.lower())])

        df = pd.DataFrame(data,
                columns='original_sentence,e1,e2,relation_type,metadata,tokenized_sentence,upos_sentence,deps_sentence,ner_sentence'.split(','))
        return df
    
    # write the dataframe into the text format accepted by the cnn model
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
                dict["relation"] = row.relation_type
                outfile.write(str(dict)+"\n")
            outfile.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input_file', default='benchmark/raw_semeval2010/TRAIN_FILE.TXT', 
        help='Input path of training examples')
    parser.add_argument('--test_input_file', default='benchmark/raw_semeval2010/TEST_FILE_FULL.TXT', 
        help='Input path of training examples')
    parser.add_argument('--output_path', default='benchmark/semeval2010/original', 
        help='Input path of training examples')
    parser.add_argument('--nlp_tool', default='stanza', choices=config.NLP_TOOLS,
        help='NLP tool name')
    parser.add_argument('--nlp_tool_type', default='general', choices=config.NLP_MODEL,
        help='NLP tool type name')
    
    args = parser.parse_args()
    
    converter = ConverterSemEval2018(args.nlp_tool, args.nlp_tool_type)
    
    converter.write_split_dataframes('benchmark/semeval20181-1/original', 'benchmark/raw_semeval20181-1/Train/', 'benchmark/raw_semeval20181-1/Test/')
    converter.write_split_dataframes('benchmark/semeval20181-2/original', 'benchmark/raw_semeval20181-2/Train/', 'benchmark/raw_semeval20181-2/Test/')