import pytest

from deepref.dataset.preprocessor.semeval2018_preprocessor import SemEval2018Preprocessor


@pytest.fixture
def preprocessor():
    return SemEval2018Preprocessor(dataset_name='semeval20181-1', nlp_tool=None)


class TestGetEntityPairs:
    def test_parses_normal_relation(self, preprocessor, tmp_path):
        (tmp_path / "relations.txt").write_text("causes(e1.1,e2.1)\n")

        result = preprocessor.get_entity_pairs(str(tmp_path))

        assert 'e1.1' in result
        assert result['e1.1']['relation'] == 'causes'
        assert result['e1.1']['e1'] == 'e1.1'
        assert result['e1.1']['e2'] == 'e2.1'

    def test_parses_reverse_relation(self, preprocessor, tmp_path):
        (tmp_path / "relations.txt").write_text("causes(e2.1,e1.1,REVERSE)\n")

        result = preprocessor.get_entity_pairs(str(tmp_path))

        assert 'e1.1' in result
        assert result['e1.1']['relation'] == 'causes'
        assert result['e1.1']['e1'] == 'e1.1'
        assert result['e1.1']['e2'] == 'e2.1'

    def test_parses_multiple_relations(self, preprocessor, tmp_path):
        (tmp_path / "relations.txt").write_text(
            "causes(e1.1,e2.1)\n"
            "prevents(e3.1,e4.1)\n"
        )

        result = preprocessor.get_entity_pairs(str(tmp_path))

        assert 'e1.1' in result
        assert 'e3.1' in result

    def test_empty_directory_returns_empty_dict(self, preprocessor, tmp_path):
        result = preprocessor.get_entity_pairs(str(tmp_path))
        assert result == {}

    def test_normal_and_reverse_in_same_file(self, preprocessor, tmp_path):
        (tmp_path / "relations.txt").write_text(
            "causes(e1.1,e2.1)\n"
            "prevents(e4.1,e3.1,REVERSE)\n"
        )

        result = preprocessor.get_entity_pairs(str(tmp_path))

        assert result['e1.1']['relation'] == 'causes'
        assert result['e3.1']['relation'] == 'prevents'
