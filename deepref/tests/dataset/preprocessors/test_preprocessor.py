import pytest

from deepref.dataset.preprocessors.preprocessor import Preprocessor
from deepref.dataset.sentence import Sentence

from deepref.tests.dataset.preprocessors.conftest import make_sentence


class TestProcessSentence:
    def setup_method(self):
        self.preprocessor = Preprocessor()

    def test_empty_indexes_leaves_sentence_unchanged(self, base_sentence):
        result = self.preprocessor.process_sentence(base_sentence, [])

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york", "yesterday"]
        assert result.entity1 == {"name": "john smith", "position": [1, 3]}
        assert result.entity2 == {"name": "new york", "position": [4, 6]}

    def test_remove_token_before_both_entities(self, base_sentence):
        # Remove "the" (index 0) — shifts both entity positions by 1
        result = self.preprocessor.process_sentence(base_sentence, [0])

        assert result.original_sentence == ["john", "smith", "visited", "new", "york", "yesterday"]
        assert result.entity1["position"] == [0, 2]
        assert result.entity2["position"] == [3, 5]

    def test_remove_token_between_entities(self, base_sentence):
        # Remove "visited" (index 3) — only entity2 position shifts by 1
        result = self.preprocessor.process_sentence(base_sentence, [3])

        assert result.original_sentence == ["the", "john", "smith", "new", "york", "yesterday"]
        assert result.entity1["position"] == [1, 3]
        assert result.entity2["position"] == [3, 5]

    def test_remove_token_after_both_entities(self, base_sentence):
        # Remove "yesterday" (index 6) — no position shifts
        result = self.preprocessor.process_sentence(base_sentence, [6])

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york"]
        assert result.entity1["position"] == [1, 3]
        assert result.entity2["position"] == [4, 6]

    def test_remove_multiple_tokens(self, base_sentence):
        # Remove "the" (0), "visited" (3), "yesterday" (6)
        # offset1: {0} < 1 → 1 ; offset2: {0,3} < 4 → 2
        result = self.preprocessor.process_sentence(base_sentence, [0, 3, 6])

        assert result.original_sentence == ["john", "smith", "new", "york"]
        assert result.entity1["position"] == [0, 2]
        assert result.entity2["position"] == [2, 4]

    def test_parallel_fields_stay_in_sync(self, base_sentence):
        result = self.preprocessor.process_sentence(base_sentence, [0, 6])

        length = len(result.original_sentence)
        assert len(result.pos_tags) == length
        assert len(result.dependencies_labels) == length
        assert len(result.ner) == length

    def test_returns_the_same_sentence_object(self, base_sentence):
        result = self.preprocessor.process_sentence(base_sentence, [])
        assert result is base_sentence

    def test_entity_names_match_tokens_after_removal(self, base_sentence):
        result = self.preprocessor.process_sentence(base_sentence, [0, 3])

        e1, e2 = result.entity1, result.entity2
        assert " ".join(result.original_sentence[e1["position"][0]:e1["position"][1]]) == e1["name"]
        assert " ".join(result.original_sentence[e2["position"][0]:e2["position"][1]]) == e2["name"]


class TestEntityIndexes:
    def setup_method(self):
        self.preprocessor = Preprocessor()

    def test_returns_union_of_both_entity_ranges(self, base_sentence):
        # entity1: [1,3) = {1,2}  entity2: [4,6) = {4,5}
        assert self.preprocessor._entity_indexes(base_sentence) == {1, 2, 4, 5}

    def test_single_token_entities(self):
        sentence = make_sentence(
            original_sentence="alice loves bob",
            entity1="{'name': 'alice', 'position': [0, 1]}",
            entity2="{'name': 'bob', 'position': [2, 3]}",
            pos_tags="NNP VBZ NNP",
            ner="PER O PER",
            dependencies_labels="nsubj ROOT dobj",
        )
        assert self.preprocessor._entity_indexes(sentence) == {0, 2}


class TestPreprocessDataset:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Preprocessor().preprocess_dataset()
