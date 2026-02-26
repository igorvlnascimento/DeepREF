from deepref.dataset.preprocessors.brackets_or_parenthesis_preprocessor import BracketsPreprocessor

from deepref.tests.dataset.preprocessors.conftest import make_sentence


class TestRemoveBracketsOrParenthesis:
    def setup_method(self):
        self.preprocessor = BracketsPreprocessor()

    def test_no_brackets_leaves_sentence_unchanged(self, base_sentence):
        result = self.preprocessor.remove_brackets_or_parenthesis(base_sentence)

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york", "yesterday"]
        assert result.entity1["position"] == [1, 3]
        assert result.entity2["position"] == [4, 6]

    def test_parentheses_outside_entities_are_removed(self):
        # "the ( former ) john smith visited new york"
        sentence = make_sentence(
            original_sentence="the ( former ) john smith visited new york",
            entity1="{'name': 'john smith', 'position': [4, 6]}",
            entity2="{'name': 'new york', 'position': [7, 9]}",
            pos_tags="DT PUNCT NN PUNCT NNP NNP VBD NNP NNP",
            ner="O O O O PER PER O LOC LOC",
            dependencies_labels="det punct appos punct nsubj nsubj ROOT compound dobj",
        )
        result = self.preprocessor.remove_brackets_or_parenthesis(sentence)

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york"]
        assert result.entity1["position"] == [1, 3]
        assert result.entity2["position"] == [4, 6]

    def test_square_brackets_outside_entities_are_removed(self):
        # "the [ former ] john smith visited new york"
        sentence = make_sentence(
            original_sentence="the [ former ] john smith visited new york",
            entity1="{'name': 'john smith', 'position': [4, 6]}",
            entity2="{'name': 'new york', 'position': [7, 9]}",
            pos_tags="DT PUNCT NN PUNCT NNP NNP VBD NNP NNP",
            ner="O O O O PER PER O LOC LOC",
            dependencies_labels="det punct appos punct nsubj nsubj ROOT compound dobj",
        )
        result = self.preprocessor.remove_brackets_or_parenthesis(sentence)

        assert result.original_sentence == ["the", "john", "smith", "visited", "new", "york"]

    def test_entity_tokens_inside_brackets_are_preserved(self):
        # "( john smith ) visited new york" — brackets around entity1 are removed,
        # entity tokens at [1,2] are skipped (preserved)
        sentence = make_sentence(
            original_sentence="( john smith ) visited new york",
            entity1="{'name': 'john smith', 'position': [1, 3]}",
            entity2="{'name': 'new york', 'position': [5, 7]}",
            pos_tags="PUNCT NNP NNP PUNCT VBD NNP NNP",
            ner="O PER PER O O LOC LOC",
            dependencies_labels="punct nsubj nsubj punct ROOT compound dobj",
        )
        result = self.preprocessor.remove_brackets_or_parenthesis(sentence)

        # brackets at [0] and [3] are removed; entity tokens [1,2] are preserved
        assert result.original_sentence == ["john", "smith", "visited", "new", "york"]
        assert result.entity1["position"] == [0, 2]
        assert result.entity2["position"] == [3, 5]

    def test_multiple_bracket_groups_removed(self):
        # "the ( a ) john smith ( b ) new york"
        sentence = make_sentence(
            original_sentence="the ( a ) john smith ( b ) new york",
            entity1="{'name': 'john smith', 'position': [4, 6]}",
            entity2="{'name': 'new york', 'position': [9, 11]}",
            pos_tags="DT PUNCT NN PUNCT NNP NNP PUNCT NN PUNCT NNP NNP",
            ner="O O O O PER PER O O O LOC LOC",
            dependencies_labels="det punct appos punct nsubj nsubj punct appos punct compound dobj",
        )
        result = self.preprocessor.remove_brackets_or_parenthesis(sentence)

        assert result.original_sentence == ["the", "john", "smith", "new", "york"]
        assert result.entity1["position"] == [1, 3]
        assert result.entity2["position"] == [3, 5]
