from deepref.dataset.preprocessors.digit_blinding_preprocessor import DigitBlindingPreprocessor

from deepref.tests.dataset.preprocessors.conftest import make_sentence


class TestDigitBlinding:
    def setup_method(self):
        self.preprocessor = DigitBlindingPreprocessor()

    def test_num_tokens_replaced_with_digit(self):
        sentence = make_sentence(
            original_sentence="john smith born 1990 new york",
            entity1="{'name': 'john smith', 'position': [0, 2]}",
            entity2="{'name': 'new york', 'position': [4, 6]}",
            pos_tags="NNP NNP VBD NUM NNP NNP",
            ner="PER PER O O LOC LOC",
            dependencies_labels="nsubj nsubj ROOT dobj compound dobj",
        )
        result = self.preprocessor.digit_blinding(sentence)

        assert result.original_sentence[3] == "DIGIT"
        assert result.original_sentence == ["john", "smith", "born", "DIGIT", "new", "york"]

    def test_non_num_tokens_unchanged(self, base_sentence):
        result = self.preprocessor.digit_blinding(base_sentence)

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york", "yesterday"]

    def test_multiple_num_tokens_all_replaced(self):
        sentence = make_sentence(
            original_sentence="john smith visited 10 times in 2020 new york",
            entity1="{'name': 'john smith', 'position': [0, 2]}",
            entity2="{'name': 'new york', 'position': [7, 9]}",
            pos_tags="NNP NNP VBD NUM NNS IN NUM NNP NNP",
            ner="PER PER O O O O O LOC LOC",
            dependencies_labels="nsubj nsubj ROOT nummod dobj prep pobj compound dobj",
        )
        result = self.preprocessor.digit_blinding(sentence)

        assert result.original_sentence[3] == "DIGIT"
        assert result.original_sentence[6] == "DIGIT"

    def test_num_at_entity_position_is_also_replaced(self):
        # DigitBlinding does not skip entity positions — it blinds all NUM tokens
        sentence = make_sentence(
            original_sentence="the 42 visited new york",
            entity1="{'name': '42', 'position': [1, 2]}",
            entity2="{'name': 'new york', 'position': [3, 5]}",
            pos_tags="DT NUM VBD NNP NNP",
            ner="O NUM O LOC LOC",
            dependencies_labels="det nsubj ROOT compound dobj",
        )
        result = self.preprocessor.digit_blinding(sentence)

        assert result.original_sentence[1] == "DIGIT"

    def test_returns_same_sentence_object(self, base_sentence):
        result = self.preprocessor.digit_blinding(base_sentence)
        assert result is base_sentence
