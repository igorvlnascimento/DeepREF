from deepref.dataset.preprocessors.entity_blinding_preprocessor import EntityBlindingPreprocessor

from deepref.tests.dataset.preprocessors.conftest import make_sentence


def _ner_sentence():
    # entity1 "john smith" (2 tokens) before entity2 "new york" (2 tokens)
    return make_sentence(
        original_sentence="the john smith visited new york yesterday",
        entity1="{'name': 'john smith', 'position': [1, 3]}",
        entity2="{'name': 'new york', 'position': [4, 6]}",
        pos_tags="DT NNP NNP VBD NNP NNP NN",
        ner="O PER PER O LOC LOC O",
    )


def _reversed_sentence():
    # entity2 "new york" (2 tokens) before entity1 "john smith" (2 tokens)
    return make_sentence(
        original_sentence="the new york visited john smith yesterday",
        entity1="{'name': 'john smith', 'position': [4, 6]}",
        entity2="{'name': 'new york', 'position': [1, 3]}",
        pos_tags="DT NNP NNP VBD NNP NNP NN",
        ner="O LOC LOC O PER PER O",
    )


class TestEntityBlinding:
    def test_entity_type_replaces_tokens_with_placeholder(self):
        preprocessor = EntityBlindingPreprocessor(entity_replacement='ENTITY', type='entity')
        result = preprocessor.entity_blinding(_ner_sentence())

        assert result.original_sentence == ["the", "ENTITY", "visited", "ENTITY", "yesterday"]
        assert result.entity1["position"] == [1, 2]
        assert result.entity2["position"] == [3, 4]

    def test_ner_type_replaces_tokens_with_ner_label(self):
        preprocessor = EntityBlindingPreprocessor(type='ner')
        result = preprocessor.entity_blinding(_ner_sentence())

        # ner[1]="PER" for entity1, ner[4]="LOC" for entity2
        assert result.original_sentence == ["the", "PER", "visited", "LOC", "yesterday"]
        assert result.entity1["position"] == [1, 2]
        assert result.entity2["position"] == [3, 4]

    def test_entity2_before_entity1_in_sentence(self):
        # else branch: entity2 comes first in the sentence
        preprocessor = EntityBlindingPreprocessor(entity_replacement='ENTITY', type='entity')
        result = preprocessor.entity_blinding(_reversed_sentence())

        assert result.original_sentence == ["the", "ENTITY", "visited", "ENTITY", "yesterday"]
        assert result.entity2["position"] == [1, 2]
        assert result.entity1["position"] == [3, 4]

    def test_entity_positions_point_to_replacement_token(self):
        preprocessor = EntityBlindingPreprocessor(entity_replacement='ENT', type='entity')
        result = preprocessor.entity_blinding(_ner_sentence())

        e1_pos = result.entity1['position'][0]
        e2_pos = result.entity2['position'][0]
        assert result.original_sentence[e1_pos] == 'ENT'
        assert result.original_sentence[e2_pos] == 'ENT'

    def test_single_token_entities(self):
        # entity1 and entity2 are each 1 token — span adjustment is 0
        sentence = make_sentence(
            original_sentence="alice loves bob",
            entity1="{'name': 'alice', 'position': [0, 1]}",
            entity2="{'name': 'bob', 'position': [2, 3]}",
            pos_tags="NNP VBZ NNP",
            ner="PER O PER",
            dependencies_labels="nsubj ROOT dobj",
        )
        preprocessor = EntityBlindingPreprocessor(entity_replacement='ENTITY', type='entity')
        result = preprocessor.entity_blinding(sentence)

        assert result.original_sentence == ["ENTITY", "loves", "ENTITY"]
        assert result.entity1["position"] == [0, 1]
        assert result.entity2["position"] == [2, 3]
