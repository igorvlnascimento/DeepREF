import pytest

from deepref.encoder.word_sentence_encoder import WordSentenceEncoder
from deepref.framework.word_embedding_loader import WordEmbeddingLoader
from deepref.module.nn.cnn import CNN
from deepref.module.nn.crcnn import CRCNN
from deepref.module.nn.sentence_gru import SentenceGRU
from deepref.module.nn.sentence_lstm import SentenceLSTM

RECURSIVE_CLASSIFIERS = [SentenceGRU, SentenceLSTM]


@pytest.fixture
def glove_sentence_encoded():
    item = {'token': ['the', 'most', 'common', 'audits', 'were', 'about', 'waste', 'and', 'recycling', '.'], 'h': {'name': 'audits', 'pos': [3, 4]}, 't': {'name': 'waste', 'pos': [6, 7]},}
    word2id, word2vec = WordEmbeddingLoader("glove").load_embedding()
    encoder = WordSentenceEncoder(token2id=word2id, word2vec=word2vec)
    tokens, pos1, pos2 = encoder.tokenize(item)
    return encoder(tokens, pos1, pos2)

def test_word_sentence_encoder(glove_sentence_encoded):
    assert glove_sentence_encoded.shape == (1, 128, 60)

def test_word_sentence_encoder_and_cnn_integration(glove_sentence_encoded):
    x = CNN(input_size=glove_sentence_encoded.shape[-1])(glove_sentence_encoded)
    
    assert x.shape == (1, 32768)

def test_word_sentence_encoder_and_crcnn_integration(glove_sentence_encoded):
    x = CRCNN(input_size=glove_sentence_encoded.shape[-1])(glove_sentence_encoded)
    
    assert x.shape == (1, 230)

@pytest.mark.parametrize("recursive_classifier", RECURSIVE_CLASSIFIERS)
def test_word_sentence_encoder_and_gru_integration(recursive_classifier, glove_sentence_encoded):
    x = recursive_classifier(input_size=glove_sentence_encoded.shape[-1])(glove_sentence_encoded)
    
    assert x.shape == (1, 256)