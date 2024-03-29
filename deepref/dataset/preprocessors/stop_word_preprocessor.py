import nltk
from nltk.corpus import stopwords

from tqdm import tqdm

from deepref.dataset.sentence import Sentence
from deepref.dataset.preprocessors.preprocessor import Preprocessor

nltk.download('stopwords')

class StopWordPreprocessor(Preprocessor):
    def __init__(self, dataset, preprocessing_types, entity_replacement=None):
        super(StopWordPreprocessor, self).__init__(dataset, preprocessing_types, entity_replacement)
        
    def preprocess_dataset(self):
        for i, sentence in tqdm(enumerate(self.dataset.train_sentences)):
            self.dataset.train_sentences[i] = self.stop_words_removal(sentence)
        for i, sentence in tqdm(enumerate(self.dataset.test_sentences)):
            self.dataset.test_sentences[i] = self.stop_words_removal(sentence)
        # for i, sentence in tqdm(enumerate(self.dataset.val_sentences)):
        #     self.dataset.val_sentences[i] = self.stop_words_removal(sentence)
            
        return self.dataset
            
    def stop_words_removal(self, sentence: Sentence):
        stop_words = set(stopwords.words('english'))
        stop_words.remove('o')
        indexes = []
        entity1_indexes = list(range(sentence.entity1['position'][0], sentence.entity1['position'][1]))
        entity2_indexes = list(range(sentence.entity2['position'][0], sentence.entity2['position'][1]))
        for j, token in enumerate(sentence.original_sentence):
            if token in stop_words and j not in entity1_indexes and j not in entity2_indexes:
                indexes.append(j)
        return self.process_sentence(sentence, indexes)