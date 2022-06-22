from tqdm import tqdm

from opennre.dataset.sentence import Sentence

from opennre.dataset.preprocessors.preprocessor import Preprocessor

class EntityBlindingPreprocessor(Preprocessor):
    def __init__(self, dataset, preprocessing_types, entity_replacement, type='ner'):
        self.type = type
        super(EntityBlindingPreprocessor, self).__init__(dataset, preprocessing_types, entity_replacement)
        
    def preprocess_dataset(self):
        if self.type == 'ner':
            for i, sentence in tqdm(enumerate(self.dataset.train_sentences)):
                self.dataset.train_sentences[i] = self.entity_blinding(sentence, type='ner')
            for i, sentence in tqdm(enumerate(self.dataset.test_sentences)):
                self.dataset.test_sentences[i] = self.entity_blinding(sentence, type='ner')
            for i, sentence in tqdm(enumerate(self.dataset.val_sentences)):
                self.dataset.val_sentences[i] = self.entity_blinding(sentence, type='ner')
        elif self.type == 'entity':
            for i, sentence in tqdm(enumerate(self.dataset.train_sentences)):
                self.dataset.train_sentences[i] = self.entity_blinding(sentence)
            for i, sentence in tqdm(enumerate(self.dataset.test_sentences)):
                self.dataset.test_sentences[i] = self.entity_blinding(sentence)
            for i, sentence in tqdm(enumerate(self.dataset.val_sentences)):
                self.dataset.val_sentences[i] = self.entity_blinding(sentence)
        else:
            for i, sentence in tqdm(enumerate(self.dataset.train_sentences)):
                self.dataset.train_sentences[i] = self.entity_blinding(sentence)
            for i, sentence in tqdm(enumerate(self.dataset.test_sentences)):
                self.dataset.test_sentences[i] = self.entity_blinding(sentence)
            for i, sentence in tqdm(enumerate(self.dataset.val_sentences)):
                self.dataset.val_sentences[i] = self.entity_blinding(sentence)
                
        return self.dataset
    
    def entity_blinding(self, sentence: Sentence, type:str='entity'):
        position1 = sentence.entity1['position']
        position2 = sentence.entity2['position']
        entity_replacement = []
        if type == 'ner':
            entity_replacement = [sentence.ner[position1[0]], sentence.ner[position2[0]]]
        elif type == 'entity':
            entity_replacement = [self.entity_replacement, self.entity_replacement]
        if position1[0] < position2[0]:
            entity1_length = position1[-1] - position1[0]    
            sentence.original_sentence = sentence.original_sentence[:position1[0]] + [entity_replacement[0]] + \
                sentence.original_sentence[position1[-1]:position2[0]] + [entity_replacement[1]] + sentence.original_sentence[position2[-1]:]
            sentence.entity1['position'] = [position1[0], position1[0]+1]
            position2[0] = position2[0] - (entity1_length - 1)
            sentence.entity2['position'] = [position2[0], position2[0]+1]
        else:
            entity1_length = position2[-1] - position2[0]
            sentence.original_sentence = sentence.original_sentence[:position2[0]] + [entity_replacement[1]] + \
                sentence.original_sentence[position2[-1]:position1[0]] + [entity_replacement[0]] + sentence.original_sentence[position1[-1]:]
            sentence.entity2['position'] = [position2[0], position2[0]+1]
            position1[0] = position1[0] - (entity1_length - 1)
            sentence.entity1['position'] = [position1[0], position1[0]+1]
        assert sentence.original_sentence[sentence.entity1['position'][0]] == entity_replacement[0]
        assert sentence.original_sentence[sentence.entity2['position'][0]] == entity_replacement[1]
        return sentence