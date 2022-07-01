from tqdm import tqdm

from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor

class DigitBlindingPreprocessor(Preprocessor):
    def __init__(self, dataset, preprocessing_types, entity_replacement=None):
        super(DigitBlindingPreprocessor, self).__init__(dataset, preprocessing_types, entity_replacement)
        
    def preprocess_dataset(self):
        for i, sentence in tqdm(enumerate(self.dataset.train_sentences)):
            self.dataset.train_sentences[i] = self.digit_blinding(sentence)
        for i, sentence in tqdm(enumerate(self.dataset.test_sentences)):
            self.dataset.test_sentences[i] = self.digit_blinding(sentence)
        for i, sentence in tqdm(enumerate(self.dataset.val_sentences)):
            self.dataset.val_sentences[i] = self.digit_blinding(sentence)
            
        return self.dataset
    
    def digit_blinding(self, sentence: Sentence):
        for j, pos in enumerate(sentence.pos_tags):
            if pos == 'NUM':
                sentence.original_sentence[j] = "DIGIT"
        return sentence        
    