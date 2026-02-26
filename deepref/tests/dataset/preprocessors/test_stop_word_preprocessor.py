from deepref.dataset.preprocessors.stop_word_preprocessor import StopWordPreprocessor

from deepref.tests.dataset.preprocessors.conftest import make_sentence


class TestStopWordsRemoval:
    def setup_method(self):
        self.preprocessor = StopWordPreprocessor()

    def test_stop_words_outside_entities_are_removed(self, base_sentence):
        # "the" at index 0 is a stop word and outside both entities
        result = self.preprocessor.stop_words_removal(base_sentence)

        assert "the" not in result.original_sentence
        assert result.entity1["position"] == [0, 2]   # shifted left by 1
        assert result.entity2["position"] == [3, 5]   # shifted left by 1

    def test_stop_words_at_entity_positions_are_preserved(self):
        # entity1 token "in" is a stop word but must not be removed
        sentence = make_sentence(
            original_sentence="alice in wonderland visited the city",
            entity1="{'name': 'alice in wonderland', 'position': [0, 3]}",
            entity2="{'name': 'city', 'position': [5, 6]}",
            pos_tags="NNP IN NNP VBD DT NN",
            ner="PER PER PER O O LOC",
            dependencies_labels="nsubj prep pobj ROOT det dobj",
        )
        result = self.preprocessor.stop_words_removal(sentence)

        # "the" at index 4 (outside entities) is removed; "in" at index 1 is preserved
        assert result.original_sentence[:3] == ["alice", "in", "wonderland"]
        assert result.entity1["position"] == [0, 3]

    def test_no_stop_words_leaves_sentence_unchanged(self):
        sentence = make_sentence(
            original_sentence="john smith visited new york",
            entity1="{'name': 'john smith', 'position': [0, 2]}",
            entity2="{'name': 'new york', 'position': [3, 5]}",
            pos_tags="NNP NNP VBD NNP NNP",
            ner="PER PER O LOC LOC",
            dependencies_labels="nsubj nsubj ROOT compound dobj",
        )
        result = self.preprocessor.stop_words_removal(sentence)

        assert result.original_sentence == ["john", "smith", "visited", "new", "york"]
        assert result.entity1["position"] == [0, 2]
        assert result.entity2["position"] == [3, 5]

    def test_multiple_stop_words_all_removed(self):
        # "the", "and", "a" are all stop words
        sentence = make_sentence(
            original_sentence="the john smith and a new york",
            entity1="{'name': 'john smith', 'position': [1, 3]}",
            entity2="{'name': 'new york', 'position': [5, 7]}",
            pos_tags="DT NNP NNP CC DT NNP NNP",
            ner="O PER PER O O LOC LOC",
            dependencies_labels="det nsubj nsubj cc det compound dobj",
        )
        result = self.preprocessor.stop_words_removal(sentence)

        assert result.original_sentence == ["john", "smith", "new", "york"]
        assert result.entity1["position"] == [0, 2]
        assert result.entity2["position"] == [2, 4]

    def test_parallel_fields_stay_in_sync(self, base_sentence):
        result = self.preprocessor.stop_words_removal(base_sentence)

        length = len(result.original_sentence)
        assert len(result.pos_tags) == length
        assert len(result.dependencies_labels) == length
        assert len(result.ner) == length
