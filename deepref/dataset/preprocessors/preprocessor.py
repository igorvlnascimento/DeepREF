import argparse
from tqdm import tqdm

from deepref.dataset.dataset import Dataset
from deepref.dataset.sentence import Sentence

class Preprocessor():
    def __init__(self, dataset: Dataset, preprocessing_types: list=[], entity_replacement: str=None):
        self.dataset = dataset
        self.preprocessing_types = preprocessing_types
        self.entity_replacement = entity_replacement
        
    def preprocess_dataset(self):
        """ Preprocess dataset. """
        pass
        
    def process_sentence(self, sentence: Sentence, indexes:list):
        entity1_indexes = list(range(sentence.entity1['position'][0], sentence.entity1['position'][1]))
        entity2_indexes = list(range(sentence.entity2['position'][0], sentence.entity2['position'][1]))
        indexes_before_ent1 = 0
        indexes_before_ent2 = 0
        for i in indexes:
            if i < entity1_indexes[0]:
                indexes_before_ent1 += 1
            if i < entity2_indexes[0]:
                indexes_before_ent2 += 1
        sentence.original_sentence = [token for i, token in enumerate(sentence.original_sentence) if i not in indexes]
        sentence.pos_tags = [pos for i, pos in enumerate(sentence.pos_tags) if i not in indexes]
        sentence.dependencies_labels = [deps for i, deps in enumerate(sentence.dependencies_labels) if i not in indexes]
        sentence.ner = [ner for i, ner in enumerate(sentence.ner) if i not in indexes]
        sentence.entity1['position'][0] = sentence.entity1['position'][0] - indexes_before_ent1
        sentence.entity1['position'][1] = sentence.entity1['position'][1] - indexes_before_ent1
        sentence.entity2['position'][0] = sentence.entity2['position'][0] - indexes_before_ent2
        sentence.entity2['position'][1] = sentence.entity2['position'][1] - indexes_before_ent2
        assert " ".join(sentence.original_sentence[sentence.entity1['position'][0]:sentence.entity1['position'][1]]) == sentence.entity1['name']
        assert " ".join(sentence.original_sentence[sentence.entity2['position'][0]:sentence.entity2['position'][1]]) == sentence.entity2['name']
        assert len(sentence.original_sentence) == len(sentence.pos_tags) == len(sentence.dependencies_labels) == len(sentence.ner)
        return sentence