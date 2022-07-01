from tqdm import tqdm

from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor

class BracketsPreprocessor(Preprocessor):
    def __init__(self, dataset, preprocessing_types, entity_replacement=None):
        super(BracketsPreprocessor, self).__init__(dataset, preprocessing_types, entity_replacement)
        
    def preprocess_dataset(self):
        for i, sentence in tqdm(enumerate(self.dataset.train_sentences)):
            self.dataset.train_sentences[i] = self.remove_brackets_or_parenthesis(sentence)
        for i, sentence in tqdm(enumerate(self.dataset.test_sentences)):
            self.dataset.test_sentences[i] = self.remove_brackets_or_parenthesis(sentence)
        for i, sentence in tqdm(enumerate(self.dataset.val_sentences)):
            self.dataset.val_sentences[i] = self.remove_brackets_or_parenthesis(sentence)
            
        return self.dataset
    
    def remove_brackets_or_parenthesis(self, sentence: Sentence):
        indexes = []
        entity1_indexes = list(range(sentence.entity1['position'][0], sentence.entity1['position'][1]))
        entity2_indexes = list(range(sentence.entity2['position'][0], sentence.entity2['position'][1]))
        brackets = False
        for j, token in enumerate(sentence.original_sentence):
            if j in entity1_indexes or j in entity2_indexes:
                continue
            elif token == '(' or token == "[":
                brackets = True
                indexes.append(j)
            elif token == ")" or token == ']':
                brackets = False
                indexes.append(j)
            elif brackets:
                indexes.append(j)
        return self.process_sentence(sentence, indexes)