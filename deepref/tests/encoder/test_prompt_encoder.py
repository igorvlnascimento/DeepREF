import pytest
import torch
from deepref.encoder.relation_encoder import PromptEntityEncoder

PRETRAINED_BERT = 'bert-base-uncased'
MAX_LENGTH = 128

ITEM_TOKEN = {
    'token': ['the', 'audits', 'were', 'about', 'waste', '.'],
    'h': {'name': 'audits', 'pos': [1, 2]},
    't': {'name': 'waste', 'pos': [4, 5]},
}

# Same sentence but with tail entity appearing before head in the sentence
ITEM_TOKEN_REV = {
    'token': ['the', 'audits', 'were', 'about', 'waste', '.'],
    'h': {'name': 'waste', 'pos': [4, 5]},
    't': {'name': 'audits', 'pos': [1, 2]},
}

# 'The audits were about waste.'
# 'audits' at [4, 10], 'waste' at [22, 27]
ITEM_TEXT = {
    'text': 'The audits were about waste.',
    'h': {'name': 'audits', 'pos': [4, 10]},
    't': {'name': 'waste', 'pos': [22, 27]},
}


@pytest.fixture(scope='module')
def encoder():
    return PromptEntityEncoder(max_length=MAX_LENGTH, pretrain_path=PRETRAINED_BERT)


def test_hidden_size(encoder):
    assert encoder.hidden_size == encoder.bert.config.hidden_size * 3


def test_special_tokens_registered(encoder):
    vocab = encoder.tokenizer.get_vocab()
    for token in ['<h>', '</h>', '<t>', '</t>']:
        assert token in vocab


def test_tokenize_token_input_shapes(encoder):
    indexed_tokens, att_mask, pos_h, pos_t, pos_mask = encoder.tokenize(ITEM_TOKEN)

    assert indexed_tokens.shape == (1, MAX_LENGTH)
    assert att_mask.shape == (1, MAX_LENGTH)
    assert pos_h.shape == (1, 1)
    assert pos_t.shape == (1, 1)
    assert pos_mask.shape == (1, 1)


def test_tokenize_text_input_shapes(encoder):
    indexed_tokens, att_mask, pos_h, pos_t, pos_mask = encoder.tokenize(ITEM_TEXT)

    assert indexed_tokens.shape == (1, MAX_LENGTH)
    assert att_mask.shape == (1, MAX_LENGTH)
    assert pos_h.shape == (1, 1)
    assert pos_t.shape == (1, 1)
    assert pos_mask.shape == (1, 1)


def test_tokenize_pos_h_points_to_h_marker(encoder):
    indexed_tokens, att_mask, pos_h, pos_t, pos_mask = encoder.tokenize(ITEM_TOKEN)

    h_id = encoder.tokenizer.convert_tokens_to_ids('<h>')
    assert indexed_tokens[0, pos_h[0, 0].item()].item() == h_id


def test_tokenize_pos_t_points_to_t_marker(encoder):
    indexed_tokens, att_mask, pos_h, pos_t, pos_mask = encoder.tokenize(ITEM_TOKEN)

    t_id = encoder.tokenizer.convert_tokens_to_ids('<t>')
    assert indexed_tokens[0, pos_t[0, 0].item()].item() == t_id


def test_tokenize_pos_mask_points_to_mask_token(encoder):
    indexed_tokens, att_mask, pos_h, pos_t, pos_mask = encoder.tokenize(ITEM_TOKEN)

    mask_id = encoder.tokenizer.mask_token_id
    assert indexed_tokens[0, pos_mask[0, 0].item()].item() == mask_id


def test_tokenize_reversed_entity_order(encoder):
    """When tail appears before head in the sentence, <t> marker should come before <h>."""
    _, _, pos_h, pos_t, _ = encoder.tokenize(ITEM_TOKEN_REV)

    assert pos_t[0, 0].item() < pos_h[0, 0].item()


def test_tokenize_attention_mask_covers_real_tokens(encoder):
    indexed_tokens, att_mask, _, _, _ = encoder.tokenize(ITEM_TOKEN)

    pad_id = encoder.tokenizer.pad_token_id
    # All positions with att_mask=0 should be padding
    padding_positions = (att_mask[0] == 0).nonzero(as_tuple=True)[0]
    for idx in padding_positions:
        assert indexed_tokens[0, idx].item() == pad_id


def test_tokenize_no_blank_padding_variable_length():
    enc = PromptEntityEncoder(max_length=MAX_LENGTH, pretrain_path=PRETRAINED_BERT, blank_padding=False)
    indexed_tokens, att_mask, _, _, _ = enc.tokenize(ITEM_TOKEN)

    # Without padding, length should be shorter than max_length for a short sentence
    assert indexed_tokens.shape[1] <= MAX_LENGTH


def test_forward_output_shape(encoder):
    indexed_tokens, att_mask, pos_h, pos_t, pos_mask = encoder.tokenize(ITEM_TOKEN)

    output = encoder(indexed_tokens, att_mask, pos_h, pos_t, pos_mask)

    hidden_size = encoder.bert.config.hidden_size
    assert output.shape == (1, hidden_size * 3)
