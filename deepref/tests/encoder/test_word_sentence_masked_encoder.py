import pytest

from deepref.encoder.word_sentence_masked_encoder import WordSentenceMaskedEncoder
from deepref.framework.word_embedding_loader import WordEmbeddingLoader
from deepref.module.nn.pcnn import PCNN


@pytest.fixture
def glove_sentence_encoded_mask():
    item = {'token': ['the', 'most', 'common', 'audits', 'were', 'about', 'waste', 'and', 'recycling', '.'], 'h': {'name': 'audits', 'pos': [3, 4]}, 't': {'name': 'waste', 'pos': [6, 7]},}
    word2id, word2vec = WordEmbeddingLoader("glove").load_embedding()
    encoder = WordSentenceMaskedEncoder(token2id=word2id, word2vec=word2vec)
    tokens, pos1, pos2, mask = encoder.tokenize(item)
    return encoder(tokens, pos1, pos2, mask)

def test_word_sentence_encoder_with_mask(glove_sentence_encoded_mask):
    sentence_encoded, mask = glove_sentence_encoded_mask
    assert sentence_encoded.shape == (1, 128, 60)
    assert mask.shape == (1, 128)

def test_word_sentence_encoder_and_pcnn_integration(glove_sentence_encoded_mask):
    sentence_encoded, mask = glove_sentence_encoded_mask

    x = PCNN(input_size=sentence_encoded.shape[-1])(sentence_encoded, mask)
    
    assert x.shape == (1, 690)
    assert mask.shape == (1, 128)