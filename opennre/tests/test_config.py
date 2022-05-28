import pytest
from opennre import config

@pytest.fixture
def preprocessing_combination():
    return [[], ['sw'], ['d'], ['b'], ['p'], ['eb'], ['nb'], ['sw', 'd'], ['sw', 'b'], ['sw', 'p'], ['sw', 'eb'], ['sw', 'nb'], ['d', 'b'], ['d', 'p'], ['d', 'eb'], ['d', 'nb'], ['b', 'p'], 
            ['b', 'eb'], ['b', 'nb'], ['p', 'eb'], ['p', 'nb'], ['sw', 'd', 'b'], ['sw', 'd', 'p'], ['sw', 'd', 'eb'], ['sw', 'd', 'nb'], ['sw', 'b', 'p'], ['sw', 'b', 'eb'], 
            ['sw', 'b', 'nb'], ['sw', 'p', 'eb'], ['sw', 'p', 'nb'], ['d', 'b', 'p'], ['d', 'b', 'eb'], ['d', 'b', 'nb'], ['d', 'p', 'eb'], ['d', 'p', 'nb'], ['b', 'p', 'eb'], ['b', 'p', 'nb'],
            ['sw', 'd', 'b', 'p'], ['sw', 'd', 'b', 'eb'], ['sw', 'd', 'b', 'nb'], ['sw', 'd', 'p', 'eb'], ['sw', 'd', 'p', 'nb'], ['sw', 'b', 'p', 'eb'], ['sw', 'b', 'p', 'nb'], 
            ['d', 'b', 'p', 'eb'], ['d', 'b', 'p', 'nb'], ['sw', 'd', 'b', 'p', 'eb'], ['sw', 'd', 'b', 'p', 'nb']]

@pytest.fixture    
def embeddings_combination():
    return ['', 'position', 'sk', 'pos_tags', 'deps', 'position sk', 'position pos_tags', 'position deps', 'sk pos_tags', 'sk deps', 'pos_tags deps', 
            'position sk pos_tags', 'position sk deps', 'position pos_tags deps', 'sk pos_tags deps', 'position sk pos_tags deps']

def test_should_return_correct_preprocessing_combinations_while_combining(preprocessing_combination):
    preprocess_combination = config.combine(config.PREPROCESSING_TYPES, 'preprocessing')
    assert len(preprocess_combination) == len(preprocessing_combination)
    assert preprocess_combination == preprocessing_combination
    
def test_should_return_correct_embeddings_combinations_while_combining(embeddings_combination):
    embed_combination = config.combine(config.TYPE_EMBEDDINGS)
    assert len(embed_combination) == len(embeddings_combination)
    assert set(embed_combination) == set(embeddings_combination)
    
def test_should_contain_correct_embeddings_keys_for_config_hparams():
    assert "sk_embed" in config.HPARAMS and "pos_tags_embed" in config.HPARAMS and "position_embed" in config.HPARAMS and "deps_embed" in config.HPARAMS