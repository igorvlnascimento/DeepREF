from deepref.dataset.preprocessors.punctuation_preprocessor import PunctuationPreprocessor

from deepref.tests.dataset.preprocessors.conftest import make_sentence


class TestRemovePunctuation:
    def setup_method(self):
        self.preprocessor = PunctuationPreprocessor()

    def test_no_punctuation_leaves_sentence_unchanged(self, base_sentence):
        result = self.preprocessor.remove_punctuation(base_sentence)

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york", "yesterday"]
        assert result.entity1["position"] == [1, 3]
        assert result.entity2["position"] == [4, 6]

    def test_punct_outside_entities_is_removed(self):
        # comma at index 3 and period at index 7 are PUNCT outside entities
        sentence = make_sentence(
            original_sentence="the john smith , visited new york .",
            entity1="{'name': 'john smith', 'position': [1, 3]}",
            entity2="{'name': 'new york', 'position': [5, 7]}",
            pos_tags="DT NNP NNP PUNCT VBD NNP NNP PUNCT",
            ner="O PER PER O O LOC LOC O",
            dependencies_labels="det nsubj nsubj punct ROOT compound dobj punct",
        )
        result = self.preprocessor.remove_punctuation(sentence)

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york"]
        assert result.entity1["position"] == [1, 3]
        assert result.entity2["position"] == [4, 6]

    def test_punct_before_entities_shifts_positions(self):
        # comma at index 0 (before both entities)
        sentence = make_sentence(
            original_sentence=", john smith visited new york",
            entity1="{'name': 'john smith', 'position': [1, 3]}",
            entity2="{'name': 'new york', 'position': [4, 6]}",
            pos_tags="PUNCT NNP NNP VBD NNP NNP",
            ner="O PER PER O LOC LOC",
            dependencies_labels="punct nsubj nsubj ROOT compound dobj",
        )
        result = self.preprocessor.remove_punctuation(sentence)

        assert result.original_sentence == ["john", "smith", "visited", "new", "york"]
        assert result.entity1["position"] == [0, 2]
        assert result.entity2["position"] == [3, 5]

    def test_punct_at_entity_position_is_preserved(self):
        # pos_tags marks entity1 tokens as PUNCT — they must not be removed
        sentence = make_sentence(
            original_sentence=". . visited new york",
            entity1="{'name': '. .', 'position': [0, 2]}",
            entity2="{'name': 'new york', 'position': [3, 5]}",
            pos_tags="PUNCT PUNCT VBD NNP NNP",
            ner="O O O LOC LOC",
            dependencies_labels="punct punct ROOT compound dobj",
        )
        result = self.preprocessor.remove_punctuation(sentence)

        # entity1 tokens [0,1] are skipped; no other PUNCT in sentence
        assert result.original_sentence == [".", ".", "visited", "new", "york"]
        assert result.entity1["position"] == [0, 2]

    def test_parallel_fields_stay_in_sync(self):
        sentence = make_sentence(
            original_sentence="the john smith , visited new york .",
            entity1="{'name': 'john smith', 'position': [1, 3]}",
            entity2="{'name': 'new york', 'position': [5, 7]}",
            pos_tags="DT NNP NNP PUNCT VBD NNP NNP PUNCT",
            ner="O PER PER O O LOC LOC O",
            dependencies_labels="det nsubj nsubj punct ROOT compound dobj punct",
        )
        result = self.preprocessor.remove_punctuation(sentence)

        length = len(result.original_sentence)
        assert len(result.pos_tags) == length
        assert len(result.dependencies_labels) == length
        assert len(result.ner) == length
