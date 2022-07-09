import os

from deepref.utils.spacy_nlp_tool import SpacyNLPTool
from deepref.utils.stanza_nlp_tool import StanzaNLPTool

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence

from deepref.dataset.semeval2010_dataset import SemEval2010Dataset
from deepref.dataset.semeval20181_dataset import SemEval20181Dataset
from deepref.dataset.semeval20182_dataset import SemEval20182Dataset
from deepref.dataset.ddi_dataset import DDIDataset

class DatasetConverter():
    
    def __init__(self, dataset_name, nlp_tool="spacy", nlp_model=None):
        
        self.dataset_name = dataset_name
        
        if nlp_tool == "spacy":
            self.nlp_tool = SpacyNLPTool(nlp_model)
        elif nlp_tool == "stanza":
            self.nlp_tool = StanzaNLPTool(nlp_model)
        else:
            self.nlp_tool = SpacyNLPTool()
                
        os.makedirs(os.path.join('benchmark', self.dataset_name, 'original'), exist_ok=True)
        
    def remove_whitespace(self, line):
        return str(" ".join(line.split()).strip())
    
    def parse_position(self, position):
        positions = position.split('-')
        return int(positions[0]), int(positions[1])
        
    # given position dictionary, sort the positions from ascending order. Assumes no overlap. 
    # will be messed up if there is overlap
    # can also check for overlap but not right now
    def sort_position_keys(self, position_dict):
        positions = list(position_dict.keys())
        sorted_positions = sorted(positions, key=lambda x: int(x.split('-')[0]))
        return sorted_positions
        
    # given the metadata, get the individual positions in the sentence and know what to replace them by
    def create_positions_dict(self, e1, e2, other_entities):
        position_dict = {}
        for pos in e1['charOffset']:
            if pos not in position_dict:
                position_dict[pos] = {'start': 'ENTITYSTART', 'end': 'ENTITYEND'}
        for pos in e2['charOffset']:
            if pos not in position_dict:
                position_dict[pos] = {'start': 'ENTITYOTHERSTART', 'end': 'ENTITYOTHEREND'}
        for other_ent in other_entities:
            for pos in other_ent['charOffset']:
                if pos not in position_dict:
                    position_dict[pos] = {'start': 'ENTITYUNRELATEDSTART', 'end': 'ENTITYUNRELATEDEND'}
        return position_dict
        
    def get_other_entities(self, entity_dict, e1, e2):
        blacklisted_set = [e1, e2]
        return [value for key, value in entity_dict.items() if key not in blacklisted_set]
    
    def tag_sentence(self, sentence, e1_data, e2_data, other_entities):
        position_dict = self.create_positions_dict(e1_data, e2_data, other_entities)
        sorted_positions = self.sort_position_keys(position_dict)
        tagged_sentence = ''
        for i in range(len(sorted_positions)):
            curr_pos = sorted_positions[i]
            curr_start_pos, curr_end_pos = self.parse_position(curr_pos)
            if i == 0:
                tagged_sentence += sentence[:curr_start_pos] + ' ' + position_dict[curr_pos]['start'] + ' ' + \
                        sentence[curr_start_pos: curr_end_pos+1] + ' ' + position_dict[curr_pos]['end'] + ' '
            else:
                prev_pos = sorted_positions[i-1]
                _, prev_end_pos = self.parse_position(prev_pos)
                middle = sentence[prev_end_pos+1 : curr_start_pos]
                if middle == '':
                    middle = ' '
                tagged_sentence += middle + ' ' + position_dict[curr_pos]['start'] + ' ' + \
                        sentence[curr_start_pos: curr_end_pos+1] + ' ' + position_dict[curr_pos]['end'] + ' '
                if i == len(sorted_positions) - 1 and curr_end_pos < len(sentence) - 1:
                    tagged_sentence += ' ' + sentence[curr_end_pos+1:]
        tagged_sentence = self.remove_whitespace(tagged_sentence)
        
        return tagged_sentence
    
    def get_entity_dict(self, *args):
        """ Get the dictionary of entity information """
        pass
    
    def get_sentences(self, path):
        """ Create a generator function that returns each tagged sentence and the relation related """
        pass
        
    def create_dataset(self, train_sentences:str, test_sentences:str) -> Dataset:
        """ Generate dataset """
        
        train_sentences_processed = [Sentence(tagged_sentence, relation, self.nlp_tool) for tagged_sentence, relation in train_sentences]
        test_sentences_processed = [Sentence(tagged_sentence, relation, self.nlp_tool) for tagged_sentence, relation in test_sentences]
        
        if self.dataset_name == "semeval2010":
            dataset = SemEval2010Dataset(self.dataset_name, train_sentences_processed, test_sentences_processed)
        elif self.dataset_name == "semeval20181-1":
            dataset = SemEval20181Dataset(self.dataset_name, train_sentences_processed, test_sentences_processed)
        elif self.dataset_name == "semeval20181-2":
            dataset = SemEval20182Dataset(self.dataset_name, train_sentences_processed, test_sentences_processed)
        elif self.dataset_name == "ddi":
            dataset = DDIDataset(self.dataset_name, train_sentences_processed, test_sentences_processed)
        
        dataset.write_dataframe()
        dataset.write_text([])
        dataset.write_classes_json()
        return dataset