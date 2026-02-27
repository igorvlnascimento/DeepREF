import pytest

from deepref.dataset.preprocessor.semeval2010_preprocessor import SemEval2010Preprocessor


@pytest.fixture
def preprocessor():
    return SemEval2010Preprocessor(nlp_tool=None)


class TestTagSentence:
    def test_replaces_e1_and_e2_tags(self, preprocessor):
        line = "1\tThe <e1>dog</e1> ate the <e2>cat</e2>."
        result = preprocessor.tag_sentence(line)
        assert "ENTITYSTART dog ENTITYEND" in result
        assert "ENTITYOTHERSTART cat ENTITYOTHEREND" in result

    def test_strips_surrounding_double_quotes(self, preprocessor):
        line = '1\t"The <e1>dog</e1> ate the <e2>cat</e2>."'
        result = preprocessor.tag_sentence(line)
        assert not result.startswith('"')
        assert not result.endswith('"')

    def test_no_quotes_leaves_content_intact(self, preprocessor):
        line = "1\tThe <e1>dog</e1> ate the <e2>cat</e2>."
        result = preprocessor.tag_sentence(line)
        assert result.startswith("The")

    def test_normalises_whitespace(self, preprocessor):
        line = "1\tThe  <e1>dog</e1>  ate  the  <e2>cat</e2>."
        result = preprocessor.tag_sentence(line)
        assert "  " not in result

    def test_original_entity_tags_are_removed(self, preprocessor):
        line = "1\tThe <e1>dog</e1> ate the <e2>cat</e2>."
        result = preprocessor.tag_sentence(line)
        assert "<e1>" not in result
        assert "</e1>" not in result
        assert "<e2>" not in result
        assert "</e2>" not in result

    def test_entity_words_are_preserved(self, preprocessor):
        line = "1\tThe <e1>dog</e1> ate the <e2>cat</e2>."
        result = preprocessor.tag_sentence(line)
        assert "dog" in result
        assert "cat" in result

    def test_multi_token_entities(self, preprocessor):
        line = "1\t<e1>John Smith</e1> visited <e2>New York</e2>."
        result = preprocessor.tag_sentence(line)
        assert "ENTITYSTART John Smith ENTITYEND" in result
        assert "ENTITYOTHERSTART New York ENTITYOTHEREND" in result


class TestGetSentences:
    def test_yields_tagged_sentence_and_relation(self, preprocessor, tmp_path):
        content = (
            '1\t"The <e1>dog</e1> ate the <e2>cat</e2>."\n'
            "Cause-Effect(e1,e2)\n"
            "Comment: no comment\n"
            "\n"
        )
        filepath = tmp_path / "train.txt"
        filepath.write_text(content)

        results = list(preprocessor.get_sentences(str(filepath)))

        assert len(results) == 1
        tagged, relation = results[0]
        assert "ENTITYSTART" in tagged
        assert relation == "Cause-Effect(e1,e2)"

    def test_yields_multiple_sentences(self, preprocessor, tmp_path):
        content = (
            '1\t"The <e1>dog</e1> ate the <e2>cat</e2>."\n'
            "Cause-Effect(e1,e2)\n"
            "Comment: none\n"
            "\n"
            '2\t"<e1>Alice</e1> loves <e2>Bob</e2>."\n'
            "Other(e1,e2)\n"
            "Comment: none\n"
            "\n"
        )
        filepath = tmp_path / "train.txt"
        filepath.write_text(content)

        results = list(preprocessor.get_sentences(str(filepath)))
        assert len(results) == 2
