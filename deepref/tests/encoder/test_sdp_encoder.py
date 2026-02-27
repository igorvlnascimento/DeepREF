"""
Tests for SDPEncoder — a dual-mode sentence encoder based on the
Shortest Dependency Path (SDP) between two marked entities.

Encoding modes:
  (1) One-hot / multi-hot bag-of-words over dependency relation labels.
  (2) Dense embedding produced by verbalizing the SDP and passing the
      resulting sentence through a SentenceEncoder.

SpaCy model used for dependency parsing: en_core_web_trf.

Test strategy
-------------
* Fixtures that need only the parser (spaCy) use a mock for the
  SentenceEncoder so we avoid loading a large LLM for every test.
* Integration tests that require a real embedding model are marked with
  ``@pytest.mark.integration`` and use SmolLM-135M-Instruct.

Input item format (mirrors the existing codebase convention):

    {
        'token': ['The', 'audits', 'were', 'about', 'waste', '.'],
        'h': {'name': 'audits', 'pos': [1, 2]},   # [start, end) token indices
        't': {'name': 'waste',  'pos': [4, 5]},
    }
"""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Constants shared across all tests
# ---------------------------------------------------------------------------

# "The audits were about waste ."
# SpaCy en_core_web_trf parse (standard):
#   were  → ROOT
#   audits → nsubj → were
#   about  → prep  → were
#   waste  → pobj  → about
#   The   → det   → audits
#   .     → punct → were
#
# SDP from 'audits' to 'waste':
#   audits --↑nsubj--> were --↓prep--> about --↓pobj--> waste
ITEM_SIMPLE = {
    'token': ['The', 'audits', 'were', 'about', 'waste', '.'],
    'h': {'name': 'audits', 'pos': [1, 2]},
    't': {'name': 'waste', 'pos': [4, 5]},
}

# e1 appears AFTER e2 in the token list (reversed order)
ITEM_REVERSED = {
    'token': ['The', 'waste', 'came', 'from', 'audits', '.'],
    'h': {'name': 'audits', 'pos': [4, 5]},
    't': {'name': 'waste', 'pos': [1, 2]},
}

# Multi-word entities
ITEM_MULTIWORD = {
    'token': ['The', 'cable', 'network', 'is', 'dedicated', 'to', 'science', 'fiction', '.'],
    'h': {'name': 'cable network', 'pos': [1, 3]},
    't': {'name': 'science fiction', 'pos': [6, 8]},
}

# Adjacent entities (direct dependency, shortest path = 2 nodes)
ITEM_ADJACENT = {
    'token': ['software', 'company', 'is', 'famous', '.'],
    'h': {'name': 'software', 'pos': [0, 1]},
    't': {'name': 'company', 'pos': [1, 2]},
}

MOCK_EMBED_DIM = 576
REAL_MODEL = 'HuggingFaceTB/SmolLM-135M-Instruct'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_sentence_encoder(embed_dim: int = MOCK_EMBED_DIM) -> MagicMock:
    """Return a SentenceEncoder mock that outputs a (1, embed_dim) tensor."""
    mock = MagicMock()
    mock.return_value = torch.zeros(1, embed_dim)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def encoder():
    """
    SDPEncoder with spaCy loaded and SentenceEncoder replaced by a mock.
    Covers all tests that don't require a real LLM forward pass.
    """
    from deepref.encoder.sdp_encoder import SDPEncoder
    return SDPEncoder(
        sentence_encoder=_make_mock_sentence_encoder(),
    )


@pytest.fixture(scope='module')
def encoder_real():
    """
    SDPEncoder with the real SmolLM model.
    Used only for integration / shape tests on encode_dense.
    """
    from deepref.encoder.sdp_encoder import SDPEncoder
    return SDPEncoder(sentence_encoder_model=REAL_MODEL)


# ===========================================================================
# 1. dep_to_full_name
# ===========================================================================

class TestDepToFullName:

    def test_nsubj_maps_to_nominal_subject(self, encoder):
        assert encoder.dep_to_full_name('nsubj') == 'nominal subject'

    def test_dobj_maps_to_direct_object(self, encoder):
        assert encoder.dep_to_full_name('dobj') == 'direct object'

    def test_prep_maps_to_prepositional_modifier(self, encoder):
        assert encoder.dep_to_full_name('prep') == 'prepositional modifier'

    def test_pobj_maps_to_object_of_preposition(self, encoder):
        assert encoder.dep_to_full_name('pobj') == 'object of preposition'

    def test_amod_maps_to_adjectival_modifier(self, encoder):
        assert encoder.dep_to_full_name('amod') == 'adjectival modifier'

    def test_advmod_maps_to_adverbial_modifier(self, encoder):
        assert encoder.dep_to_full_name('advmod') == 'adverbial modifier'

    def test_compound_maps_to_compound_modifier(self, encoder):
        assert encoder.dep_to_full_name('compound') == 'compound modifier'

    def test_det_maps_to_determiner(self, encoder):
        assert encoder.dep_to_full_name('det') == 'determiner'

    def test_root_maps_to_root(self, encoder):
        assert encoder.dep_to_full_name('ROOT') == 'root'

    def test_cc_maps_to_coordinating_conjunction(self, encoder):
        assert encoder.dep_to_full_name('cc') == 'coordinating conjunction'

    def test_conj_maps_to_conjunct(self, encoder):
        assert encoder.dep_to_full_name('conj') == 'conjunct'

    def test_attr_maps_to_attribute(self, encoder):
        assert encoder.dep_to_full_name('attr') == 'attribute'

    def test_acl_maps_to_clausal_modifier_of_noun(self, encoder):
        assert encoder.dep_to_full_name('acl') == 'clausal modifier of noun'

    def test_agent_maps_to_agent(self, encoder):
        assert encoder.dep_to_full_name('agent') == 'agent'

    def test_aux_maps_to_auxiliary(self, encoder):
        assert encoder.dep_to_full_name('aux') == 'auxiliary'

    def test_ccomp_maps_to_clausal_complement(self, encoder):
        assert encoder.dep_to_full_name('ccomp') == 'clausal complement'

    def test_xcomp_maps_to_open_clausal_complement(self, encoder):
        assert encoder.dep_to_full_name('xcomp') == 'open clausal complement'

    def test_poss_maps_to_possession_modifier(self, encoder):
        assert encoder.dep_to_full_name('poss') == 'possession modifier'

    def test_neg_maps_to_negation_modifier(self, encoder):
        assert encoder.dep_to_full_name('neg') == 'negation modifier'

    def test_appos_maps_to_appositional_modifier(self, encoder):
        assert encoder.dep_to_full_name('appos') == 'appositional modifier'

    def test_unknown_label_returns_itself(self, encoder):
        assert encoder.dep_to_full_name('unknowndep') == 'unknowndep'

    def test_empty_string_returns_empty_string(self, encoder):
        assert encoder.dep_to_full_name('') == ''


# ===========================================================================
# 2. extract_sdp
# ===========================================================================

class TestExtractSdp:
    """
    extract_sdp(item) → list of (token_text, dep_full_name, direction) tuples.

    Convention:
      * Each tuple i describes the i-th node in the path AND the edge that
        leads FROM node i TO node i+1.
      * The last tuple has dep_full_name=None, direction=None (terminal node).
      * direction is 'UP' (child → head) or 'DOWN' (head → child).
    """

    # --- return type ---

    def test_returns_list(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        assert isinstance(result, list)

    def test_each_element_is_three_tuple(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        for element in result:
            assert len(element) == 3

    def test_path_not_empty(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        assert len(result) >= 2  # at least e1 and e2

    # --- start / end nodes ---

    def test_first_token_matches_e1(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        first_token, _, _ = result[0]
        assert first_token == ITEM_SIMPLE['h']['name']

    def test_last_token_matches_e2(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        last_token, _, _ = result[-1]
        assert last_token == ITEM_SIMPLE['t']['name']

    def test_last_node_dep_is_none(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        _, dep, direction = result[-1]
        assert dep is None
        assert direction is None

    # --- intermediate nodes have non-None dep and direction ---

    def test_intermediate_nodes_have_dep_label(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        for token, dep, direction in result[:-1]:
            assert dep is not None, f"Node '{token}' is missing a dep label"

    def test_intermediate_nodes_have_direction(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        for token, dep, direction in result[:-1]:
            assert direction is not None, f"Node '{token}' is missing a direction"

    def test_direction_values_are_up_or_down(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        for _, _, direction in result[:-1]:
            assert direction in ('UP', 'DOWN'), f"Unexpected direction: {direction!r}"

    # --- dep labels use full names, not spaCy abbreviations ---

    _ABBREVIATED = frozenset({
        'nsubj', 'dobj', 'prep', 'pobj', 'amod', 'advmod',
        'compound', 'det', 'attr', 'acl', 'aux', 'cc', 'conj',
        'agent', 'ccomp', 'xcomp', 'poss', 'neg', 'appos', 'mark',
    })

    def test_dep_labels_are_not_abbreviated(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        for _, dep, _ in result[:-1]:
            assert dep not in self._ABBREVIATED, \
                f"Abbreviated dep label found: {dep!r}"

    # --- semantic correctness for ITEM_SIMPLE ---

    def test_simple_sentence_token_sequence(self, encoder):
        """SDP for 'audits'→'waste' should traverse: audits, were, about, waste."""
        result = encoder.extract_sdp(ITEM_SIMPLE)
        token_seq = [tok for tok, _, _ in result]
        assert token_seq == ['audits', 'were', 'about', 'waste']

    def test_simple_sentence_dep_contains_nominal_subject(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        deps = [dep for _, dep, _ in result if dep is not None]
        assert 'nominal subject' in deps

    def test_simple_sentence_dep_contains_prepositional_modifier(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        deps = [dep for _, dep, _ in result if dep is not None]
        assert 'prepositional modifier' in deps

    def test_simple_sentence_dep_contains_object_of_preposition(self, encoder):
        result = encoder.extract_sdp(ITEM_SIMPLE)
        deps = [dep for _, dep, _ in result if dep is not None]
        assert 'object of preposition' in deps

    def test_simple_sentence_first_step_direction_is_up(self, encoder):
        """
        'audits' is a child of 'were' in the tree, so the first step
        (audits → were) goes UP.
        """
        result = encoder.extract_sdp(ITEM_SIMPLE)
        _, _, direction = result[0]
        assert direction == 'UP'

    # --- reversed entity order ---

    def test_reversed_item_first_token_is_e1(self, encoder):
        result = encoder.extract_sdp(ITEM_REVERSED)
        first_token, _, _ = result[0]
        assert first_token == ITEM_REVERSED['h']['name']

    def test_reversed_item_last_token_is_e2(self, encoder):
        result = encoder.extract_sdp(ITEM_REVERSED)
        last_token, _, _ = result[-1]
        assert last_token == ITEM_REVERSED['t']['name']

    # --- multi-word entities ---

    def test_multiword_first_token_belongs_to_e1_span(self, encoder):
        result = encoder.extract_sdp(ITEM_MULTIWORD)
        first_token, _, _ = result[0]
        e1_tokens = set(ITEM_MULTIWORD['token'][
            ITEM_MULTIWORD['h']['pos'][0]: ITEM_MULTIWORD['h']['pos'][1]
        ])
        assert first_token in e1_tokens

    def test_multiword_last_token_belongs_to_e2_span(self, encoder):
        result = encoder.extract_sdp(ITEM_MULTIWORD)
        last_token, _, _ = result[-1]
        e2_tokens = set(ITEM_MULTIWORD['token'][
            ITEM_MULTIWORD['t']['pos'][0]: ITEM_MULTIWORD['t']['pos'][1]
        ])
        assert last_token in e2_tokens

    # --- adjacent entities ---

    def test_adjacent_entities_path_has_exactly_two_nodes(self, encoder):
        """
        'software' is a compound modifier of 'company': direct dependency,
        so the SDP should have exactly 2 nodes.
        """
        result = encoder.extract_sdp(ITEM_ADJACENT)
        assert len(result) == 2

    def test_deterministic_on_repeated_calls(self, encoder):
        result_a = encoder.extract_sdp(ITEM_SIMPLE)
        result_b = encoder.extract_sdp(ITEM_SIMPLE)
        assert result_a == result_b


# ===========================================================================
# 3. build_dep_chain
# ===========================================================================

class TestBuildDepChain:

    def test_returns_string(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        assert isinstance(chain, str)

    def test_not_empty(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        assert len(chain) > 0

    def test_contains_e1_token(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        assert ITEM_SIMPLE['h']['name'] in chain

    def test_contains_e2_token(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        assert ITEM_SIMPLE['t']['name'] in chain

    def test_contains_nominal_subject(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        assert 'nominal subject' in chain

    def test_contains_prepositional_modifier(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        assert 'prepositional modifier' in chain

    def test_contains_object_of_preposition(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        assert 'object of preposition' in chain

    def test_tokens_appear_in_path_order(self, encoder):
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        # 'audits' should appear before 'waste'
        assert chain.index('audits') < chain.index('waste')

    def test_expected_format_simple_sentence(self, encoder):
        """
        Expected chain:
            audits --nominal subject--> were --prepositional modifier-->
            about --object of preposition--> waste
        """
        path = encoder.extract_sdp(ITEM_SIMPLE)
        chain = encoder.build_dep_chain(path)
        expected = (
            'audits --nominal subject--> were '
            '--prepositional modifier--> about '
            '--object of preposition--> waste'
        )
        assert chain == expected


# ===========================================================================
# 4. verbalize
# ===========================================================================

class TestVerbalize:

    _SENTENCE = 'The audits were about waste .'
    _E1 = 'audits'
    _E2 = 'waste'
    _CHAIN = (
        'audits --nominal subject--> were '
        '--prepositional modifier--> about '
        '--object of preposition--> waste'
    )

    def test_returns_string(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert isinstance(result, str)

    def test_exact_format(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        expected = (
            f"Sentence {self._SENTENCE} | "
            f"Entity-1: [{self._E1}] | "
            f"Entity-2: [{self._E2}] | "
            f"Dependency path: {self._CHAIN}"
        )
        assert result == expected

    def test_sentence_keyword_present(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert result.startswith('Sentence ')

    def test_entity_1_in_brackets(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert f'[{self._E1}]' in result

    def test_entity_2_in_brackets(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert f'[{self._E2}]' in result

    def test_dep_chain_present(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert self._CHAIN in result

    def test_separator_bars_present(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert result.count(' | ') == 3

    def test_entity_1_label_before_entity_2_label(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert result.index('Entity-1:') < result.index('Entity-2:')

    def test_entity_labels_before_dep_path(self, encoder):
        result = encoder.verbalize(self._SENTENCE, self._E1, self._E2, self._CHAIN)
        assert result.index('Entity-2:') < result.index('Dependency path:')


# ===========================================================================
# 5. encode_onehot
# ===========================================================================

class TestEncodeOnehot:

    def test_returns_tensor(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        assert isinstance(result, torch.Tensor)

    def test_is_one_dimensional(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        assert result.ndim == 1

    def test_length_equals_vocab_size(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        assert result.shape[0] == len(encoder.dep_vocab)

    def test_dtype_is_float32(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        assert result.dtype == torch.float32

    def test_all_values_are_zero_or_one(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        unique_vals = set(result.tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_at_least_one_active_feature(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        assert result.sum().item() >= 1

    def test_nominal_subject_bit_is_set(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        idx = encoder.dep_vocab.index('nominal subject')
        assert result[idx].item() == 1.0

    def test_prepositional_modifier_bit_is_set(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        idx = encoder.dep_vocab.index('prepositional modifier')
        assert result[idx].item() == 1.0

    def test_object_of_preposition_bit_is_set(self, encoder):
        result = encoder.encode_onehot(ITEM_SIMPLE)
        idx = encoder.dep_vocab.index('object of preposition')
        assert result[idx].item() == 1.0

    def test_irrelevant_dep_bit_is_zero(self, encoder):
        """'direct object' is not part of the audits→waste SDP."""
        result = encoder.encode_onehot(ITEM_SIMPLE)
        if 'direct object' in encoder.dep_vocab:
            idx = encoder.dep_vocab.index('direct object')
            assert result[idx].item() == 0.0

    def test_consistent_on_repeated_calls(self, encoder):
        r1 = encoder.encode_onehot(ITEM_SIMPLE)
        r2 = encoder.encode_onehot(ITEM_SIMPLE)
        assert torch.equal(r1, r2)

    def test_multiword_entities_produce_valid_vector(self, encoder):
        result = encoder.encode_onehot(ITEM_MULTIWORD)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 1
        assert set(result.tolist()).issubset({0.0, 1.0})

    def test_reversed_entities_produce_valid_vector(self, encoder):
        result = encoder.encode_onehot(ITEM_REVERSED)
        assert result.sum().item() >= 1


# ===========================================================================
# 6. encode_dense
# ===========================================================================

class TestEncodeDenseMocked:
    """Tests that exercise encode_dense logic without loading a real LLM."""

    def test_returns_tensor(self, encoder):
        result = encoder.encode_dense(ITEM_SIMPLE)
        assert isinstance(result, torch.Tensor)

    def test_is_two_dimensional(self, encoder):
        result = encoder.encode_dense(ITEM_SIMPLE)
        assert result.ndim == 2

    def test_batch_dimension_is_one(self, encoder):
        result = encoder.encode_dense(ITEM_SIMPLE)
        assert result.shape[0] == 1

    def test_embed_dim_matches_mock(self, encoder):
        result = encoder.encode_dense(ITEM_SIMPLE)
        assert result.shape[1] == MOCK_EMBED_DIM

    def test_sentence_encoder_called_with_verbalized_string(self, encoder):
        """encode_dense must call sentence_encoder with the verbalized sentence."""
        encoder.sentence_encoder.reset_mock()
        encoder.encode_dense(ITEM_SIMPLE)
        encoder.sentence_encoder.assert_called_once()
        call_args = encoder.sentence_encoder.call_args
        # The verbalized string must be among the positional / keyword args
        all_args = list(call_args.args) + list(call_args.kwargs.values())
        verbalized = any(
            isinstance(a, str) and 'Dependency path:' in a
            for a in all_args
        )
        assert verbalized, "sentence_encoder was not called with a verbalized string"

    def test_verbalized_string_contains_entity_1(self, encoder):
        encoder.sentence_encoder.reset_mock()
        encoder.encode_dense(ITEM_SIMPLE)
        call_args = encoder.sentence_encoder.call_args
        all_args = list(call_args.args) + list(call_args.kwargs.values())
        assert any(
            isinstance(a, str) and ITEM_SIMPLE['h']['name'] in a
            for a in all_args
        )

    def test_verbalized_string_contains_entity_2(self, encoder):
        encoder.sentence_encoder.reset_mock()
        encoder.encode_dense(ITEM_SIMPLE)
        call_args = encoder.sentence_encoder.call_args
        all_args = list(call_args.args) + list(call_args.kwargs.values())
        assert any(
            isinstance(a, str) and ITEM_SIMPLE['t']['name'] in a
            for a in all_args
        )


@pytest.mark.integration
class TestEncodeDenseReal:
    """
    Integration tests using the actual SmolLM model.
    Run with: pytest -m integration
    """

    def test_output_shape_matches_model_embed_dim(self, encoder_real):
        result = encoder_real.encode_dense(ITEM_SIMPLE)
        assert result.shape == (1, MOCK_EMBED_DIM)

    def test_output_is_normalized(self, encoder_real):
        result = encoder_real.encode_dense(ITEM_SIMPLE)
        norm = torch.linalg.norm(result, dim=-1).item()
        assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm}"

    def test_deterministic_output(self, encoder_real):
        r1 = encoder_real.encode_dense(ITEM_SIMPLE)
        r2 = encoder_real.encode_dense(ITEM_SIMPLE)
        assert torch.allclose(r1, r2)

    def test_different_items_produce_different_embeddings(self, encoder_real):
        r1 = encoder_real.encode_dense(ITEM_SIMPLE)
        r2 = encoder_real.encode_dense(ITEM_REVERSED)
        assert not torch.equal(r1, r2)

    def test_multiword_entities(self, encoder_real):
        result = encoder_real.encode_dense(ITEM_MULTIWORD)
        assert result.shape == (1, MOCK_EMBED_DIM)
