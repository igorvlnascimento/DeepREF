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
from unittest import mock
from unittest.mock import MagicMock

from deepref.encoder.llm_encoder import LLMEncoder

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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def encoder():
    """
    BoWSDPEncoder with spaCy loaded.
    Covers dep label mapping, SDP extraction, dep chain, verbalize, and encode_onehot tests.
    No transformer model is needed.
    """
    from deepref.encoder.sdp_encoder import BoWSDPEncoder
    return BoWSDPEncoder()


@pytest.fixture(scope='module')
def encoder_verbalized():
    """
    VerbalizedSDPEncoder with spaCy loaded and forward() replaced by a mock.
    Covers tokenize and encode_dense tests that don't require a real LLM.
    """
    from deepref.encoder.sdp_encoder import VerbalizedSDPEncoder
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    with mock.patch('deepref.utils.model_registry.AutoModel.from_pretrained', return_value=mock_model), \
         mock.patch('deepref.utils.model_registry.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        enc = VerbalizedSDPEncoder()
    enc.forward = MagicMock(return_value=torch.zeros(1, MOCK_EMBED_DIM))
    return enc


@pytest.fixture(scope='module')
def encoder_real():
    """
    VerbalizedSDPEncoder with the real SmolLM model.
    Used only for integration / shape tests on encode_dense.
    """
    from deepref.encoder.sdp_encoder import VerbalizedSDPEncoder
    return VerbalizedSDPEncoder(model_name=REAL_MODEL)


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

    _SENTENCE = ("[E1] ALBA [/E1] – the Bolivarian Alternative for the Americas – "
                 "was founded by Venezuelan President Hugo Chavez and Cuban leader "
                 "[E2] Fidel Castro [/E2] in 2004 and also includes Bolivia , "
                 "Nicaragua and the Caribbean island of Dominica .")
    _E1 = 'ALBA'
    _E2 = 'Fidel Castro'
    _CHAIN = (
        f'[ENTITY_1: {_E1}] --nsubj:pass(UP)--> founded/VERB '
        f'--obl(DOWN)--> Castro/PROPN --flat(DOWN)--> [ENTITY_2: {_E2}]'
    )

    def test_returns_string(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        assert isinstance(result, str)

    def test_exact_format(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        expected = (
            f"Instruct: Given a syntactic dependency path between two named entities, identify the semantic relation they hold.\n"
            f"Query: {self._CHAIN}"
        )
        assert result == expected

    def test_entity_1_in_brackets(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        assert f'{self._E1}' in result

    def test_entity_2_in_brackets(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        assert f'{self._E2}' in result

    def test_dep_chain_present(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        assert self._CHAIN in result

    def test_entity_1_label_before_entity_2_label(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        assert result.index('ENTITY_1:') < result.index('ENTITY_2:')

    def test_entity_labels1_before_entity_names(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        assert result.index('ENTITY_1:') < result.index(self._E1)

    def test_entity_labels2_before_entity_names(self, encoder):
        result = encoder.verbalize(self._CHAIN)
        assert result.index('ENTITY_2:') < result.index(self._E2)


# ===========================================================================
# 5a. BoWSDPEncoder.tokenize
# ===========================================================================

class TestBoWTokenize:
    """BoWSDPEncoder.tokenize returns the raw SDP path (same as extract_sdp)."""

    def test_returns_list(self, encoder):
        result = encoder.tokenize(ITEM_SIMPLE)
        assert isinstance(result, list)

    def test_each_element_is_three_tuple(self, encoder):
        result = encoder.tokenize(ITEM_SIMPLE)
        for element in result:
            assert len(element) == 3

    def test_same_result_as_extract_sdp(self, encoder):
        assert encoder.tokenize(ITEM_SIMPLE) == encoder.extract_sdp(ITEM_SIMPLE)


# ===========================================================================
# 5b. encode_onehot
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

    def test_encode_onehot_delegates_to_forward(self, encoder):
        """encode_onehot must return the same tensor as forward."""
        assert torch.equal(encoder.encode_onehot(ITEM_SIMPLE), encoder.forward(ITEM_SIMPLE))


# ===========================================================================
# 6. tokenize
# ===========================================================================

class TestTokenize:
    """Tests for VerbalizedSDPEncoder.tokenize — verbalizes the SDP and tokenizes it."""

    _FAKE_TOKENS = {
        'input_ids': torch.zeros(1, 10, dtype=torch.long),
        'attention_mask': torch.ones(1, 10, dtype=torch.long),
    }

    def test_returns_dict(self, encoder_verbalized):
        with mock.patch.object(LLMEncoder, 'tokenize_prompt', return_value=self._FAKE_TOKENS):
            result = encoder_verbalized.tokenize(ITEM_SIMPLE)
        assert isinstance(result, dict)

    def test_has_input_ids(self, encoder_verbalized):
        with mock.patch.object(LLMEncoder, 'tokenize_prompt', return_value=self._FAKE_TOKENS):
            result = encoder_verbalized.tokenize(ITEM_SIMPLE)
        assert 'input_ids' in result

    def test_has_attention_mask(self, encoder_verbalized):
        with mock.patch.object(LLMEncoder, 'tokenize_prompt', return_value=self._FAKE_TOKENS):
            result = encoder_verbalized.tokenize(ITEM_SIMPLE)
        assert 'attention_mask' in result

    def test_verbalized_string_contains_entity_1(self, encoder_verbalized):
        """tokenize must verbalize using e1's name."""
        with mock.patch.object(LLMEncoder, 'tokenize_prompt', return_value=self._FAKE_TOKENS) as mock_tok:
            encoder_verbalized.tokenize(ITEM_SIMPLE)
        _, verbalized_text = mock_tok.call_args[0]
        assert ITEM_SIMPLE['h']['name'] in verbalized_text

    def test_verbalized_string_contains_entity_2(self, encoder_verbalized):
        """tokenize must verbalize using e2's name."""
        with mock.patch.object(LLMEncoder, 'tokenize_prompt', return_value=self._FAKE_TOKENS) as mock_tok:
            encoder_verbalized.tokenize(ITEM_SIMPLE)
        _, verbalized_text = mock_tok.call_args[0]
        assert ITEM_SIMPLE['t']['name'] in verbalized_text

    def test_verbalized_string_contains_query_keyword(self, encoder_verbalized):
        with mock.patch.object(LLMEncoder, 'tokenize_prompt', return_value=self._FAKE_TOKENS) as mock_tok:
            encoder_verbalized.tokenize(ITEM_SIMPLE)
        _, verbalized_text = mock_tok.call_args[0]
        assert 'Query:' in verbalized_text


# ===========================================================================
# 7. encode_dense
# ===========================================================================

class TestEncodeDenseMocked:
    """Tests that exercise encode_dense logic without loading a real LLM."""

    def test_returns_tensor(self, encoder_verbalized):
        result = encoder_verbalized.encode_dense(ITEM_SIMPLE)
        assert isinstance(result, torch.Tensor)

    def test_is_two_dimensional(self, encoder_verbalized):
        result = encoder_verbalized.encode_dense(ITEM_SIMPLE)
        assert result.ndim == 2

    def test_batch_dimension_is_one(self, encoder_verbalized):
        result = encoder_verbalized.encode_dense(ITEM_SIMPLE)
        assert result.shape[0] == 1

    def test_embed_dim_matches_mock(self, encoder_verbalized):
        result = encoder_verbalized.encode_dense(ITEM_SIMPLE)
        assert result.shape[1] == MOCK_EMBED_DIM

    def test_forward_called_with_item(self, encoder_verbalized):
        """encode_dense must delegate to forward with the original item dict."""
        encoder_verbalized.forward.reset_mock()
        encoder_verbalized.encode_dense(ITEM_SIMPLE)
        encoder_verbalized.forward.assert_called_once_with(ITEM_SIMPLE)

    def test_forward_called_once_per_encode_dense_call(self, encoder_verbalized):
        encoder_verbalized.forward.reset_mock()
        encoder_verbalized.encode_dense(ITEM_SIMPLE)
        assert encoder_verbalized.forward.call_count == 1


@pytest.mark.integration
class TestEncodeDenseReal:
    """
    Integration tests using the actual SmolLM model.
    Run with: pytest -m integration
    """

    _FAKE_TOKENS = {
        'input_ids': torch.zeros(1, MOCK_EMBED_DIM, dtype=torch.long),
        'attention_mask': torch.ones(1, MOCK_EMBED_DIM, dtype=torch.long),
    }

    def test_output_shape_matches_model_embed_dim(self, encoder_real):
        result = encoder_real.encode_dense(ITEM_SIMPLE)
        assert result.shape == (1, MOCK_EMBED_DIM)

    def test_output_is_normalized(self, encoder_real):
        result = encoder_real.encode_dense(ITEM_SIMPLE)
        norm = torch.linalg.norm(result, dim=1).item()
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


# ===========================================================================
# 8. Arc.__str__
# ===========================================================================

class TestArc:

    def test_str_up_direction(self):
        from deepref.encoder.sdp_encoder import Arc
        arc = Arc(deprel='nsubj', direction='UP')
        assert str(arc) == '--nsubj(UP)-->'

    def test_str_down_direction(self):
        from deepref.encoder.sdp_encoder import Arc
        arc = Arc(deprel='prep', direction='DOWN')
        assert str(arc) == '--prep(DOWN)-->'

    def test_str_with_colon_in_deprel(self):
        from deepref.encoder.sdp_encoder import Arc
        arc = Arc(deprel='nsubj:pass', direction='UP')
        assert str(arc) == '--nsubj:pass(UP)-->'

    def test_str_contains_deprel(self):
        from deepref.encoder.sdp_encoder import Arc
        arc = Arc(deprel='pobj', direction='DOWN')
        assert 'pobj' in str(arc)

    def test_str_contains_direction(self):
        from deepref.encoder.sdp_encoder import Arc
        arc = Arc(deprel='pobj', direction='DOWN')
        assert 'DOWN' in str(arc)


# ===========================================================================
# 9. SDPNode.verbalize
# ===========================================================================

class TestSDPNodeVerbalize:

    def test_entity_1_node(self):
        from deepref.encoder.sdp_encoder import SDPNode
        node = SDPNode(
            token_idx=0, text='ALBA', upos='PROPN',
            is_entity=True, entity_role='ENTITY_1', entity_span='ALBA',
        )
        assert node.verbalize() == '[ENTITY_1: ALBA]'

    def test_entity_2_node_multiword(self):
        from deepref.encoder.sdp_encoder import SDPNode
        node = SDPNode(
            token_idx=5, text='Castro', upos='PROPN',
            is_entity=True, entity_role='ENTITY_2', entity_span='Fidel Castro',
        )
        assert node.verbalize() == '[ENTITY_2: Fidel Castro]'

    def test_intermediate_node_no_off_path(self):
        from deepref.encoder.sdp_encoder import SDPNode
        node = SDPNode(
            token_idx=2, text='founded', upos='VERB',
            is_entity=False,
        )
        assert node.verbalize() == 'founded/VERB'

    def test_intermediate_node_single_off_path_token(self):
        from deepref.encoder.sdp_encoder import SDPNode
        node = SDPNode(
            token_idx=4, text='Castro', upos='PROPN',
            is_entity=False, off_path_tokens=['Fidel'],
        )
        assert node.verbalize() == 'Castro[+Fidel]/PROPN'

    def test_intermediate_node_multiple_off_path_tokens(self):
        from deepref.encoder.sdp_encoder import SDPNode
        node = SDPNode(
            token_idx=4, text='built', upos='VERB',
            is_entity=False, off_path_tokens=['was', 'recently'],
        )
        assert node.verbalize() == 'built[+was+recently]/VERB'

    def test_entity_node_ignores_off_path_tokens(self):
        from deepref.encoder.sdp_encoder import SDPNode
        node = SDPNode(
            token_idx=0, text='ALBA', upos='PROPN',
            is_entity=True, entity_role='ENTITY_1', entity_span='ALBA',
            off_path_tokens=['some', 'neighbor'],
        )
        # off_path_tokens are ignored for entity nodes
        assert node.verbalize() == '[ENTITY_1: ALBA]'


# ===========================================================================
# 10. extract_entities_and_clean
# ===========================================================================

class TestExtractEntitiesAndClean:

    _MARKED = '[E1] ALBA [/E1] was founded by [E2] Fidel Castro [/E2] in 2004 .'

    def test_returns_three_element_tuple(self, encoder):
        result = encoder.extract_entities_and_clean(self._MARKED)
        assert isinstance(result, tuple) and len(result) == 3

    def test_e1_text_extracted(self, encoder):
        _, e1, _ = encoder.extract_entities_and_clean(self._MARKED)
        assert e1 == 'ALBA'

    def test_e2_text_extracted(self, encoder):
        _, _, e2 = encoder.extract_entities_and_clean(self._MARKED)
        assert e2 == 'Fidel Castro'

    def test_clean_sentence_has_no_e1_markers(self, encoder):
        clean, _, _ = encoder.extract_entities_and_clean(self._MARKED)
        assert '[E1]' not in clean and '[/E1]' not in clean

    def test_clean_sentence_has_no_e2_markers(self, encoder):
        clean, _, _ = encoder.extract_entities_and_clean(self._MARKED)
        assert '[E2]' not in clean and '[/E2]' not in clean

    def test_clean_sentence_preserves_entity_text(self, encoder):
        clean, _, _ = encoder.extract_entities_and_clean(self._MARKED)
        assert 'ALBA' in clean
        assert 'Fidel Castro' in clean

    def test_clean_sentence_preserves_context_words(self, encoder):
        clean, _, _ = encoder.extract_entities_and_clean(self._MARKED)
        assert 'founded' in clean

    def test_missing_both_markers_raises_value_error(self, encoder):
        with pytest.raises(ValueError):
            encoder.extract_entities_and_clean('no entity markers here')

    def test_missing_e2_marker_raises_value_error(self, encoder):
        with pytest.raises(ValueError):
            encoder.extract_entities_and_clean('[E1] ALBA [/E1] no e2 marker')

    def test_extra_whitespace_in_span_is_normalized(self, encoder):
        marked = '[E1]  multi  word  [/E1] verb [E2] other [/E2]'
        _, e1, _ = encoder.extract_entities_and_clean(marked)
        assert e1 == 'multi word'


# ===========================================================================
# 11. find_sdp  (BFS on adjacency dict)
# ===========================================================================

# Linear chain: 0 -- 1 -- 2 -- 3
_LINEAR_ADJ = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

# Tree: center=0, leaves=1,2; 3 is child of 1
_TREE_ADJ = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}


class TestFindSdp:

    def test_src_equals_tgt_returns_singleton(self, encoder):
        assert encoder.find_sdp(_LINEAR_ADJ, 0, 0) == [0]

    def test_direct_neighbors_path_length_two(self, encoder):
        result = encoder.find_sdp(_LINEAR_ADJ, 0, 1)
        assert result == [0, 1]

    def test_linear_chain_full_path(self, encoder):
        result = encoder.find_sdp(_LINEAR_ADJ, 0, 3)
        assert result == [0, 1, 2, 3]

    def test_tree_path_through_root(self, encoder):
        """Path 3→2 must go 3→1→0→2 (only route in the tree)."""
        result = encoder.find_sdp(_TREE_ADJ, 3, 2)
        assert result == [3, 1, 0, 2]

    def test_disconnected_graph_returns_none(self, encoder):
        adj = {0: [1], 1: [0], 2: [3], 3: [2]}   # two components
        assert encoder.find_sdp(adj, 0, 2) is None

    def test_result_starts_at_src(self, encoder):
        result = encoder.find_sdp(_LINEAR_ADJ, 2, 0)
        assert result[0] == 2

    def test_result_ends_at_tgt(self, encoder):
        result = encoder.find_sdp(_LINEAR_ADJ, 2, 0)
        assert result[-1] == 0

    def test_reverse_direction_gives_reversed_path(self, encoder):
        fwd = encoder.find_sdp(_LINEAR_ADJ, 0, 3)
        rev = encoder.find_sdp(_LINEAR_ADJ, 3, 0)
        assert fwd == list(reversed(rev))


# ===========================================================================
# 12. path_centric_pruning  (K-hop on adjacency dict)
# ===========================================================================

# Star: center=0, leaves=1,2,3,4
_STAR_ADJ = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}


class TestPathCentricPruning:

    def test_k0_returns_only_sdp_nodes(self, encoder):
        sdp = [1, 0, 2]
        result = encoder.path_centric_pruning(_STAR_ADJ, sdp, K=0)
        assert result == {0, 1, 2}

    def test_k0_does_not_include_off_path_leaves(self, encoder):
        sdp = [1, 0, 2]
        result = encoder.path_centric_pruning(_STAR_ADJ, sdp, K=0)
        assert 3 not in result and 4 not in result

    def test_k1_adds_direct_neighbors_of_sdp(self, encoder):
        sdp = [1, 0, 2]
        result = encoder.path_centric_pruning(_STAR_ADJ, sdp, K=1)
        # Leaves 3 and 4 are 1 hop from center (on SDP)
        assert {0, 1, 2, 3, 4}.issubset(result)

    def test_large_k_reaches_entire_connected_component(self, encoder):
        sdp = [0, 1]
        result = encoder.path_centric_pruning(_LINEAR_ADJ, sdp, K=100)
        assert result == {0, 1, 2, 3}

    def test_sdp_nodes_always_retained_regardless_of_k(self, encoder):
        sdp = [0, 1]
        for k in range(3):
            result = encoder.path_centric_pruning(_LINEAR_ADJ, sdp, k)
            assert {0, 1}.issubset(result)

    def test_single_node_sdp_k0(self, encoder):
        result = encoder.path_centric_pruning(_STAR_ADJ, [0], K=0)
        assert result == {0}


# ===========================================================================
# Fixture: synthetic Sentence for graph / pruning tests
# ===========================================================================
#
# Dependency tree:
#          were(0) [ROOT]
#         /    |    \
#   audits(2) about(3) .(5)
#   [nsubj]  [prep]   [punct]
#      |         |
#    The(1)   waste(4)
#    [det]    [pobj]
#
# SDP from audits(2) to waste(4):  2 → 0 → 3 → 4
# Off-path tokens:  The(1), .(5)
# LCA: were(0)

@pytest.fixture(scope='module')
def sentence_fixture():
    from deepref.nlp.nlp_tool import Token, Sentence
    tokens = [
        Token(i=0, text='were',   dep_='ROOT',  head_i=0, pos_='VERB'),
        Token(i=1, text='The',    dep_='det',   head_i=2, pos_='DET'),
        Token(i=2, text='audits', dep_='nsubj', head_i=0, pos_='NOUN'),
        Token(i=3, text='about',  dep_='prep',  head_i=0, pos_='ADP'),
        Token(i=4, text='waste',  dep_='pobj',  head_i=3, pos_='NOUN'),
        Token(i=5, text='.',      dep_='punct', head_i=0, pos_='PUNCT'),
    ]
    return Sentence(tokens=tokens, subj_span=(2, 2), obj_span=(4, 4))


# ===========================================================================
# 13. build_dep_graph
# ===========================================================================

class TestBuildDepGraph:

    def test_returns_networkx_graph(self, encoder, sentence_fixture):
        import networkx as nx
        G = encoder.build_dep_graph(sentence_fixture)
        assert isinstance(G, nx.Graph)

    def test_node_count_equals_token_count(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        assert len(G.nodes) == len(sentence_fixture.tokens)

    def test_root_self_loop_not_added_as_edge(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        assert not G.has_edge(0, 0)

    def test_nsubj_edge_exists(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        assert G.has_edge(2, 0)   # audits --nsubj--> were

    def test_pobj_edge_exists(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        assert G.has_edge(4, 3)   # waste --pobj--> about

    def test_det_edge_exists(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        assert G.has_edge(1, 2)   # The --det--> audits

    def test_edge_carries_dep_label(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        assert G.edges[2, 0]['dep'] == 'nsubj'

    def test_graph_is_undirected(self, encoder, sentence_fixture):
        import networkx as nx
        G = encoder.build_dep_graph(sentence_fixture)
        assert not isinstance(G, nx.DiGraph)

    def test_edge_count_equals_non_root_tokens(self, encoder, sentence_fixture):
        """One edge per token except the ROOT (which self-loops and is skipped)."""
        G = encoder.build_dep_graph(sentence_fixture)
        non_root = sum(1 for t in sentence_fixture.tokens if t.i != t.head_i)
        assert len(G.edges) == non_root


# ===========================================================================
# 14. find_entity_head
# ===========================================================================

class TestFindEntityHead:

    def test_single_token_subj_returns_own_index(self, encoder, sentence_fixture):
        # audits(2).head_i=0 is outside span (2,2)
        assert encoder.find_entity_head(sentence_fixture, (2, 2)) == 2

    def test_single_token_obj_returns_own_index(self, encoder, sentence_fixture):
        # waste(4).head_i=3 is outside span (4,4)
        assert encoder.find_entity_head(sentence_fixture, (4, 4)) == 4

    def test_multitoken_returns_syntactic_head(self, encoder, sentence_fixture):
        # Span (1, 2) = [The, audits].
        # The(1).head_i=2 — inside span → not head.
        # audits(2).head_i=0 — outside span → this is the head.
        assert encoder.find_entity_head(sentence_fixture, (1, 2)) == 2

    def test_root_token_span_returns_root(self, encoder, sentence_fixture):
        # were(0).head_i=0 == own index; fallback returns span[0]
        assert encoder.find_entity_head(sentence_fixture, (0, 0)) == 0


# ===========================================================================
# 15. get_ancestors
# ===========================================================================

class TestGetAncestors:

    def test_leaf_to_root_path(self, encoder, sentence_fixture):
        # waste(4) → about(3) → were(0) [root]
        result = encoder.get_ancestors(sentence_fixture, 4)
        assert result == [4, 3, 0]

    def test_root_node_returns_only_root(self, encoder, sentence_fixture):
        result = encoder.get_ancestors(sentence_fixture, 0)
        assert result == [0]

    def test_direct_child_of_root(self, encoder, sentence_fixture):
        # audits(2) → were(0)
        result = encoder.get_ancestors(sentence_fixture, 2)
        assert result == [2, 0]

    def test_result_starts_at_given_token(self, encoder, sentence_fixture):
        result = encoder.get_ancestors(sentence_fixture, 4)
        assert result[0] == 4

    def test_result_ends_at_root(self, encoder, sentence_fixture):
        result = encoder.get_ancestors(sentence_fixture, 4)
        root_idx = next(t.i for t in sentence_fixture.tokens if t.head_i == t.i)
        assert result[-1] == root_idx


# ===========================================================================
# 16. find_lca
# ===========================================================================

class TestFindLca:

    def test_lca_of_audits_and_waste_is_were(self, encoder, sentence_fixture):
        # ancestors of audits(2): [2, 0]; ancestors of waste(4): [4, 3, 0]
        lca = encoder.find_lca(sentence_fixture, 2, 4)
        assert lca == 0   # were

    def test_lca_of_sibling_nodes_is_parent(self, encoder, sentence_fixture):
        # audits(2) and about(3) are both children of were(0)
        lca = encoder.find_lca(sentence_fixture, 2, 3)
        assert lca == 0

    def test_lca_of_node_with_itself(self, encoder, sentence_fixture):
        lca = encoder.find_lca(sentence_fixture, 2, 2)
        assert lca == 2

    def test_lca_is_ancestor_of_both_inputs(self, encoder, sentence_fixture):
        lca = encoder.find_lca(sentence_fixture, 1, 4)
        anc1 = encoder.get_ancestors(sentence_fixture, 1)
        anc2 = encoder.get_ancestors(sentence_fixture, 4)
        assert lca in anc1 and lca in anc2


# ===========================================================================
# 17. get_lca_subtree
# ===========================================================================

class TestGetLcaSubtree:

    def test_subtree_from_root_contains_all_tokens(self, encoder, sentence_fixture):
        subtree = encoder.get_lca_subtree(sentence_fixture, 0)
        all_indices = {t.i for t in sentence_fixture.tokens}
        assert subtree == all_indices

    def test_subtree_from_about_contains_about_and_waste(self, encoder, sentence_fixture):
        # about(3) has only waste(4) as child
        subtree = encoder.get_lca_subtree(sentence_fixture, 3)
        assert subtree == {3, 4}

    def test_subtree_from_leaf_contains_only_leaf(self, encoder, sentence_fixture):
        subtree = encoder.get_lca_subtree(sentence_fixture, 4)
        assert subtree == {4}

    def test_subtree_from_audits_contains_audits_and_the(self, encoder, sentence_fixture):
        # audits(2) has The(1) as child
        subtree = encoder.get_lca_subtree(sentence_fixture, 2)
        assert subtree == {2, 1}

    def test_lca_node_is_always_in_subtree(self, encoder, sentence_fixture):
        for lca in range(len(sentence_fixture.tokens)):
            subtree = encoder.get_lca_subtree(sentence_fixture, lca)
            assert lca in subtree


# ===========================================================================
# 18. find_shortest_dep_path
# ===========================================================================

class TestFindShortestDepPath:

    def test_finds_sdp_from_audits_to_waste(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        path = encoder.find_shortest_dep_path(G, 2, 4)
        assert path == [2, 0, 3, 4]

    def test_adjacent_nodes_path_has_length_two(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        path = encoder.find_shortest_dep_path(G, 2, 0)
        assert path == [2, 0]

    def test_disconnected_nodes_return_fallback(self, encoder):
        import networkx as nx
        G = nx.Graph()
        G.add_node(0)
        G.add_node(5)   # no edge between them
        path = encoder.find_shortest_dep_path(G, 0, 5)
        assert path == [0, 5]

    def test_path_starts_at_subj_head(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        path = encoder.find_shortest_dep_path(G, 2, 4)
        assert path[0] == 2

    def test_path_ends_at_obj_head(self, encoder, sentence_fixture):
        G = encoder.build_dep_graph(sentence_fixture)
        path = encoder.find_shortest_dep_path(G, 2, 4)
        assert path[-1] == 4


# ===========================================================================
# 19. path_centric_prune  (full Sentence-based method)
# ===========================================================================

class TestPathCentricPrune:

    def test_returns_tuple_of_three(self, encoder, sentence_fixture):
        result = encoder.path_centric_prune(sentence_fixture, K=0)
        assert len(result) == 3

    def test_k0_keeps_sdp_nodes_only(self, encoder, sentence_fixture):
        kept, sdp, _ = encoder.path_centric_prune(sentence_fixture, K=0)
        # SDP: audits(2) → were(0) → about(3) → waste(4)
        assert {0, 2, 3, 4}.issubset(kept)
        assert 1 not in kept   # The — off-path
        assert 5 not in kept   # .  — off-path

    def test_k1_adds_off_path_direct_neighbors(self, encoder, sentence_fixture):
        kept, _, _ = encoder.path_centric_prune(sentence_fixture, K=1)
        # The(1) is 1 hop from audits(2); .(5) is 1 hop from were(0)
        assert {0, 1, 2, 3, 4, 5}.issubset(kept)

    def test_sdp_returned_connects_entity_heads(self, encoder, sentence_fixture):
        _, sdp, _ = encoder.path_centric_prune(sentence_fixture, K=0)
        assert sdp[0] == 2 and sdp[-1] == 4

    def test_lca_is_were(self, encoder, sentence_fixture):
        _, _, lca = encoder.path_centric_prune(sentence_fixture, K=0)
        assert lca == 0   # were

    def test_sdp_is_list_of_ints(self, encoder, sentence_fixture):
        _, sdp, _ = encoder.path_centric_prune(sentence_fixture, K=0)
        assert isinstance(sdp, list) and all(isinstance(n, int) for n in sdp)


# ===========================================================================
# 20. build_pruned_adjacency
# ===========================================================================

class TestBuildPrunedAdjacency:

    def test_returns_dict_with_required_keys(self, encoder, sentence_fixture):
        result = encoder.build_pruned_adjacency(sentence_fixture, {0, 2, 3, 4})
        assert {'adj_list', 'adj_matrix', 'token_order', 'idx_map'}.issubset(result)

    def test_token_order_is_sorted(self, encoder, sentence_fixture):
        kept = {4, 0, 2, 3}
        result = encoder.build_pruned_adjacency(sentence_fixture, kept)
        assert result['token_order'] == sorted(kept)

    def test_adj_matrix_has_self_loops(self, encoder, sentence_fixture):
        kept = {0, 2, 3, 4}
        matrix = encoder.build_pruned_adjacency(sentence_fixture, kept)['adj_matrix']
        for i in range(len(kept)):
            assert matrix[i][i] == 1

    def test_adj_matrix_is_symmetric(self, encoder, sentence_fixture):
        kept = {0, 2, 3, 4}
        matrix = encoder.build_pruned_adjacency(sentence_fixture, kept)['adj_matrix']
        n = len(kept)
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == matrix[j][i]

    def test_adj_list_includes_audits_were_edge(self, encoder, sentence_fixture):
        result = encoder.build_pruned_adjacency(sentence_fixture, {0, 2, 3, 4})
        adj = result['adj_list']
        assert 0 in adj[2] and 2 in adj[0]

    def test_adj_list_includes_pobj_edge(self, encoder, sentence_fixture):
        result = encoder.build_pruned_adjacency(sentence_fixture, {0, 2, 3, 4})
        adj = result['adj_list']
        assert 3 in adj[4] and 4 in adj[3]

    def test_excluded_tokens_absent_from_adj_list(self, encoder, sentence_fixture):
        kept = {0, 2, 3, 4}   # excludes The(1) and .(5)
        result = encoder.build_pruned_adjacency(sentence_fixture, kept)
        assert 1 not in result['adj_list']
        assert 5 not in result['adj_list']

    def test_adj_matrix_size_equals_kept_count(self, encoder, sentence_fixture):
        kept = {0, 2, 3, 4}
        matrix = encoder.build_pruned_adjacency(sentence_fixture, kept)['adj_matrix']
        assert len(matrix) == len(kept)
        assert all(len(row) == len(kept) for row in matrix)


# ===========================================================================
# 20. VerbalizedSDPEncoder.verbalize — full pipeline, semeval2010 examples
# ===========================================================================
#
# Items derived from benchmark/semeval2010_test.csv (first rows):
#
#  ITEM_AUDITS_WASTE  — "The most common audits were about waste and recycling ."
#                        Message-Topic(e1,e2)
#  ITEM_COMPANY_CHAIRS — "The company fabricates plastic chairs ."
#                         Product-Producer(e2,e1)
#  ITEM_MASTER_STICK   — "The school master teaches the lesson with a stick ."
#                         Instrument-Agency(e2,e1)
#  ITEM_BODY_RESERVOIR — "The suspect dumped the dead body into a local reservoir ."
#                         Entity-Destination(e1,e2)
#
# mark_sentence(item) inserts [E1]/[/E1] and [E2]/[/E2] around the entity spans,
# then VerbalizedSDPEncoder.verbalize(marked_sentence) runs the full
# parse → SDP → prune → verbalize pipeline and returns:
#
#   Instruct: Given a syntactic dependency path between two named entities,
#             identify the semantic relation they hold.
#   Query: <structured dep path>

ITEM_AUDITS_WASTE = {
    'token': ['The', 'most', 'common', 'audits', 'were', 'about', 'waste', 'and', 'recycling', '.'],
    'h': {'name': 'audits', 'pos': [3, 4]},
    't': {'name': 'waste',  'pos': [6, 7]},
}

ITEM_COMPANY_CHAIRS = {
    'token': ['The', 'company', 'fabricates', 'plastic', 'chairs', '.'],
    'h': {'name': 'company', 'pos': [1, 2]},
    't': {'name': 'chairs',  'pos': [4, 5]},
}

ITEM_MASTER_STICK = {
    'token': ['The', 'school', 'master', 'teaches', 'the', 'lesson', 'with', 'a', 'stick', '.'],
    'h': {'name': 'master', 'pos': [2, 3]},
    't': {'name': 'stick',  'pos': [8, 9]},
}

ITEM_BODY_RESERVOIR = {
    'token': ['The', 'suspect', 'dumped', 'the', 'dead', 'body', 'into', 'a', 'local', 'reservoir', '.'],
    'h': {'name': 'body',       'pos': [5, 6]},
    't': {'name': 'reservoir',  'pos': [9, 10]},
}


class TestVerbalizedSDPEncoderVerbalize:
    """Tests for VerbalizedSDPEncoder.verbalize(marked_sentence).

    Uses mark_sentence to build the marked sentence from items derived
    from benchmark/semeval2010_test.csv, then passes it to verbalize
    which runs the full parse → SDP → prune → verbalize pipeline.
    """

    # ------------------------------------------------------------------
    # mark_sentence — sanity checks before calling verbalize
    # ------------------------------------------------------------------

    def test_mark_sentence_audits_waste_contains_e1_markers(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        assert '[E1]' in marked and '[/E1]' in marked

    def test_mark_sentence_audits_waste_contains_e2_markers(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        assert '[E2]' in marked and '[/E2]' in marked

    def test_mark_sentence_audits_waste_e1_wraps_audits(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        assert '[E1] audits [/E1]' in marked

    def test_mark_sentence_audits_waste_e2_wraps_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        assert '[E2] waste [/E2]' in marked

    def test_mark_sentence_company_chairs_e1_wraps_company(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_COMPANY_CHAIRS)
        assert '[E1] company [/E1]' in marked

    def test_mark_sentence_company_chairs_e2_wraps_chairs(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_COMPANY_CHAIRS)
        assert '[E2] chairs [/E2]' in marked

    def test_mark_sentence_master_stick_e1_before_e2(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_MASTER_STICK)
        assert marked.index('[E1]') < marked.index('[E2]')

    def test_mark_sentence_body_reservoir_e1_wraps_body(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_BODY_RESERVOIR)
        assert '[E1] body [/E1]' in marked

    # ------------------------------------------------------------------
    # verbalize — output structure
    # ------------------------------------------------------------------

    def test_verbalize_returns_string_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert isinstance(result, str)

    def test_verbalize_contains_instruct_prefix_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert result.startswith('Instruct:')

    def test_verbalize_contains_query_prefix_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert 'Query:' in result

    def test_verbalize_instruct_before_query_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert result.index('Instruct:') < result.index('Query:')

    def test_verbalize_entity_1_name_in_output_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert 'audits' in result

    def test_verbalize_entity_2_name_in_output_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert 'waste' in result

    def test_verbalize_returns_string_company_chairs(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_COMPANY_CHAIRS)
        result = encoder_verbalized.verbalize(marked)
        assert isinstance(result, str)

    def test_verbalize_contains_instruct_prefix_company_chairs(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_COMPANY_CHAIRS)
        result = encoder_verbalized.verbalize(marked)
        assert result.startswith('Instruct:')

    def test_verbalize_entity_1_name_in_output_company_chairs(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_COMPANY_CHAIRS)
        result = encoder_verbalized.verbalize(marked)
        assert 'company' in result

    def test_verbalize_entity_2_name_in_output_company_chairs(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_COMPANY_CHAIRS)
        result = encoder_verbalized.verbalize(marked)
        assert 'chairs' in result

    def test_verbalize_returns_string_master_stick(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_MASTER_STICK)
        result = encoder_verbalized.verbalize(marked)
        assert isinstance(result, str)

    def test_verbalize_entity_1_name_in_output_master_stick(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_MASTER_STICK)
        result = encoder_verbalized.verbalize(marked)
        assert 'master' in result

    def test_verbalize_entity_2_name_in_output_master_stick(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_MASTER_STICK)
        result = encoder_verbalized.verbalize(marked)
        assert 'stick' in result

    def test_verbalize_returns_string_body_reservoir(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_BODY_RESERVOIR)
        result = encoder_verbalized.verbalize(marked)
        assert isinstance(result, str)

    def test_verbalize_entity_1_name_in_output_body_reservoir(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_BODY_RESERVOIR)
        result = encoder_verbalized.verbalize(marked)
        assert 'body' in result

    def test_verbalize_entity_2_name_in_output_body_reservoir(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_BODY_RESERVOIR)
        result = encoder_verbalized.verbalize(marked)
        assert 'reservoir' in result

    def test_verbalize_contains_entity_label_tags_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert 'ENTITY_1:' in result and 'ENTITY_2:' in result

    def test_verbalize_entity_1_label_before_entity_2_label_audits_waste(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_AUDITS_WASTE)
        result = encoder_verbalized.verbalize(marked)
        assert result.index('ENTITY_1:') < result.index('ENTITY_2:')

    def test_verbalize_contains_entity_label_tags_body_reservoir(self, encoder_verbalized):
        marked = encoder_verbalized.mark_sentence(ITEM_BODY_RESERVOIR)
        result = encoder_verbalized.verbalize(marked)
        assert 'ENTITY_1:' in result and 'ENTITY_2:' in result
