from opennre.encoder.bert_encoder import BERTEntityEncoder
from opennre.data.generate_parser_dict import save2json, csv2id

from unittest import mock

import os
import json
import pytest

PRETRAINED_BERT = 'bert-base-uncased'

DATASET = 'semeval2010'

@pytest.fixture
def upos2id_and_deps2id():
    upos2id, deps2id = csv2id(DATASET)
    save2json(DATASET, upos2id, deps2id)
    upos2id = json.loads(open(os.path.join('opennre', 'data', f'{DATASET}_upos2id.json'), 'r').read())
    deps2id = json.loads(open(os.path.join('opennre', 'data', f'{DATASET}_deps2id.json'), 'r').read())
    return upos2id, deps2id

def test_should_return_tokens_concatenated_with_sdp_without_entities_having_only_sdp_embedding(upos2id_and_deps2id):
    upos2id, deps2id = upos2id_and_deps2id
    bert_entity = BERTEntityEncoder(PRETRAINED_BERT, upos2id, deps2id, sdp_embedding=True)
    item = {'token': ['the', 'most', 'common', 'audits', 'were', 'about', 'waste', 'and', 'recycling', '.'], 'h': {'name': 'audits', 'pos': [3, 4]}, 't': {'name': 'waste', 'pos': [6, 7]}, 'pos': ['DET', 'ADV', 'ADJ', 'NOUN', 'AUX', 'ADP', 'NOUN', 'CCONJ', 'NOUN', 'PUNCT'], 'deps': ['det', 'advmod', 'amod', 'nsubj', 'root', 'prep', 'compound', 'cc', 'conj', 'punct'], 'ner': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 'sdp': ['audits', 'were', 'about', 'waste'], 'relation': 'Message-Topic(e1,e2)'}
    result = bert_entity.tokenize(item)
    indexed_token = result[0]
    assert (indexed_token == 0).nonzero(as_tuple=True)[1][0].item() == len(item["token"]) + len(item["sdp"][1:-1]) + 7
    
def test_should_return_tokens_concatenated_with_empty_sdp_without_entities_having_only_sdp_emebdding(upos2id_and_deps2id):
    upos2id, deps2id = upos2id_and_deps2id
    bert_entity = BERTEntityEncoder(PRETRAINED_BERT, upos2id, deps2id, sdp_embedding=True)
    item = {'token': ['avian', 'influenza', 'is', 'an', 'infectious', 'disease', 'of', 'birds', 'caused', 'by', 'type', 'a', 'strains', 'of', 'the', 'influenza', 'virus', '.'], 'h': {'name': 'influenza', 'pos': [1, 2]}, 't': {'name': 'virus', 'pos': [16, 17]}, 'pos': ['ADJ', 'NOUN', 'AUX', 'DET', 'ADJ', 'NOUN', 'ADP', 'NOUN', 'VERB', 'ADP', 'NOUN', 'PRON', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'], 'deps': ['amod', 'compound', 'root', 'det', 'amod', 'attr', 'prep', 'pobj', 'acl', 'agent', 'compound', 'compound', 'pobj', 'prep', 'det', 'amod', 'pobj', 'punct'], 'ner': ['NORP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 'sdp': ['influenza', 'virus'], 'relation': 'Cause-Effect(e2,e1)'}
    result = bert_entity.tokenize(item)
    indexed_token = result[0]
    assert (indexed_token == 0).nonzero(as_tuple=True)[1][0].item() == len(item["token"]) + len(item["sdp"][1:-1]) + 7
    
def test_should_return_coorect_indexes_for_sk_entities(upos2id_and_deps2id):
    upos2id, deps2id = upos2id_and_deps2id
    bert_entity = BERTEntityEncoder(PRETRAINED_BERT, upos2id, deps2id, sdp_embedding=True)
    item = {'token': ['avian', 'influenza', 'is', 'an', 'infectious', 'disease', 'of', 'birds', 'caused', 'by', 'type', 'a', 'strains', 'of', 'the', 'influenza', 'virus', '.'], 'h': {'name': 'influenza', 'pos': [1, 2]}, 't': {'name': 'virus', 'pos': [16, 17]}, 'pos': ['ADJ', 'NOUN', 'AUX', 'DET', 'ADJ', 'NOUN', 'ADP', 'NOUN', 'VERB', 'ADP', 'NOUN', 'PRON', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'], 'deps': ['amod', 'compound', 'root', 'det', 'amod', 'attr', 'prep', 'pobj', 'acl', 'agent', 'compound', 'compound', 'pobj', 'prep', 'det', 'amod', 'pobj', 'punct'], 'ner': ['NORP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 'sdp': ['influenza', 'virus'], 'relation': 'Cause-Effect(e2,e1)'}
    result = bert_entity.tokenize(item)
    indexed_token = result[0]
    assert (indexed_token == 0).nonzero(as_tuple=True)[1][0].item() == len(item["token"]) + len(item["sdp"][1:-1]) + 7
    
    